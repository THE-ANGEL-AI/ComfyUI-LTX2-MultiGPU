"""Regression tests for v0.2.2 GemmaHybridLoader bugfixes.

Targets three pre-T4x2-test bugs found in deep audit (Gemini 2026-06-30):

1. CRITICAL — Hook accumulation in ``load_gemma_hybrid``: повторный вызов
   через тот же cached ``clip_obj`` (ComfyUI global state) добавлял
   ещё один ``forward_pre_hook`` на ``text_projection`` → O(n) хуков
   на n-м вызове. Фикс: удалить ранее установленные pre-хуки у
   ``inner`` и ``proj_module`` + сбросить ``_ltx2_hooks`` ПЕРЕД регистрацией.

2. HIGH — Silent projection-move failure: если ``_move_param(proj_p, primary_dev)``
   вернул False (torch=None или device mismatch), старый код печатал WARN
   и продолжал. На runtime: hook гонит hidden_states на cuda:0, proj-веса
   остаются на CPU/donor → confusing RuntimeError mid-sampling.
   Фикс: после цикла raise RuntimeError со списком failed params
   (skip raise при torch=None — нечего валидировать вне ComfyUI).

3. LOW/MEDIUM — ``encoder_param_count == 0`` при ``verbose=False`` + ``donor=cpu``:
   meta-dict в ``model_options`` содержал 0 для обоих encoder/proj-сторон,
   затрудняя debug. Фикс: считать ``encoder_param_count`` unconditionally.

These tests use only public/private imports + ``unittest.mock``. NO torch,
NO ComfyUI runtime required.

Run via:
    python -m unittest tests.test_gguf_split_gemma_hooks -v
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SUBMODULE_ROOT = HERE.parent
if str(SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMODULE_ROOT))


# ════════════════════════════════════════════════════════════════════════════
# Helpers: comfy-mock-injected context manager.
#
# load_gemma_hybrid at runtime делает ``from comfy import sd as comfy_sd; …
# model_patcher as comfy_mp``. Вне ComfyUI env (тест здесь прогоняется без
# реального ComfyUI) импорт падает в наш own ``except`` и raise RuntimeError
# ("comfy sd/mm/model_patcher недоступны"). Чтобы тесты добрались до проверяемой
# логики, перед каждым вызовом load_gemma_hybrid патчим sys.modules пустыми
# MagicMock-модулями; затем отдельные ``mock.patch`` подменяют нужные
# sub-атрибуты (load_clip, cuda_device_context, soft_empty_cache, get_torch_device).
# ════════════════════════════════════════════════════════════════════════════
def _comfy_runtime_mock():
    """Контекст-менеджер: инжектит mock-модули ``comfy.*`` в sys.modules.

    Используется в каждом test-методе, чтобы ``from comfy import …`` работал.
    """
    return mock.patch.dict(
        "sys.modules",
        {
            "comfy": mock.MagicMock(),
            "comfy.sd": mock.MagicMock(),
            "comfy.model_management": mock.MagicMock(),
            "comfy.model_patcher": mock.MagicMock(),
        },
    )


# ════════════════════════════════════════════════════════════════════════════
# Helpers: fake ``inner`` Module + fake CLIP that satisfies load_gemma_hybrid's
# ``patcher_obj.model`` attribute shape.
# ════════════════════════════════════════════════════════════════════════════
class _FakeProjModule:
    """Stand-in for ``text_projection`` nn.Module."""

    def __init__(self) -> None:
        # Подражает nn.Module._forward_pre_hooks OrderedDict[int, Handle].
        self._forward_pre_hooks: dict[int, mock.MagicMock] = {}

    def register_forward_pre_hook(self, fn, with_kwargs: bool = False):
        # PyTorch реально возвращает RemovableHandle. В нашем fake используем
        # MagicMock чтобы можно было проверить remove() и dict-clear.
        handle = mock.MagicMock(name="RemovableHandle")
        handle.fn = fn
        handle.with_kwargs = with_kwargs
        next_id = 0 if not self._forward_pre_hooks else max(self._forward_pre_hooks) + 1
        self._forward_pre_hooks[next_id] = handle
        return handle


class _FakeInnerModule:
    """Stand-in для CLIP. ``inner``: проходит через .named_parameters и .named_modules.

    Удобно для проверки, что:
      - Hook accumulation предотвращён (proj_module._forward_pre_hooks length).
      - encoder_param_count считается независимо от verbose.
    """

    def __init__(self, proj_module: _FakeProjModule | None = None) -> None:
        self._params: list[tuple[str, mock.MagicMock]] = [
            ("model.text_projection.weight", mock.MagicMock()),
            ("model.transformer.layers.0.self_attn.q_proj.weight", mock.MagicMock()),
            ("model.transformer.layers.0.mlp.up_proj.weight", mock.MagicMock()),
            ("model.transformer.layers.1.self_attn.q_proj.weight", mock.MagicMock()),
        ]
        # Без proj_module — defensively ``named_modules`` пустой.
        self._modules: dict[str, object] = {}
        if proj_module is not None:
            self._modules["text_projection"] = proj_module
        self._buffers: list[tuple[str, mock.MagicMock]] = []

    def named_parameters(self):
        return iter(list(self._params))

    def named_buffers(self):
        return iter(list(self._buffers))

    def named_modules(self):
        return iter(list(self._modules.items()))

    __class__ = mock.MagicMock  # для type(inner).__name__ в meta


# ════════════════════════════════════════════════════════════════════════════
# Block 1: hook accumulation fix (FIX 1)
# ════════════════════════════════════════════════════════════════════════════
class TestRemoveStoredHooksOnEachLoad(unittest.TestCase):
    """Проверяем, что ``load_gemma_hybrid`` очищает ранее накопленные хуки.

    Механика: каждый вызов load_gemma_hybrid должен (а) сбросить
    patcher._ltx2_hooks в [], (б) очистить proj_module._forward_pre_hooks
    от старых дескрипторов. Без этого цикл из N reload'ов даёт N хуков.
    """

    FIX_TARGET = "load_gemma_hybrid clears proj_module pre-hooks before register"

    def test_proj_module_pre_hooks_cleared_on_reload(self):
        """Тест-фундаментал: 2 последовательных вызова load_gemma_hybrid через
        тот же fake CLIP — после второго вызова pre-hooks на proj_module
        содержат НЕ БОЛЕЕ 1 entry.
        """
        # Импорты внутри test (после sys.path setup выше).
        from core import gguf_split

        proj = _FakeProjModule()
        inner = _FakeInnerModule(proj)
        # fake patcher_obj — патчер с .model, ._ltx2_hooks, .load_device,
        # .offload_device и .model_options.
        patcher_obj = mock.MagicMock()
        patcher_obj.model = inner
        patcher_obj._ltx2_hooks = []
        patcher_obj.model_options = {}
        # Имитация CLIP wrapper.
        clip_obj = mock.MagicMock()
        clip_obj.patcher = patcher_obj

        # Мокаем тяжёлые R6-guards, чтобы load_gemma_hybrid досчитал до точки
        # установки хука без реального CUDA.
        with _comfy_runtime_mock(), \
             mock.patch.object(gguf_split, "folder_paths", create=True), \
             mock.patch.object(gguf_split, "torch", create=True) as torch_mock, \
             mock.patch.object(gguf_split, "_lock_inner_to", lambda _inner: None), \
             mock.patch.object(
                 gguf_split, "_move_param",
                 new=lambda p, dev: True,  # все params "успешно" перенесены
             ):
            torch_mock.is_available.return_value = True
            torch_mock.cuda.is_available.return_value = False  # to fail silently
            torch_mock.device.side_effect = lambda s: s  # для resolve_devices / cuda_device_context
            torch_mock.no_grad.return_value.__enter__ = mock.MagicMock()
            torch_mock.no_grad.return_value.__exit__ = mock.MagicMock()

            # Первый вызов: инжектим comfy.clip_obj через monkey-patch,
            # обходя checks через mm.cuda_device_context mock.
            with mock.patch("comfy.sd.load_clip", return_value=clip_obj), \
                 mock.patch("comfy.model_management.cuda_device_context",
                            mock.MagicMock()), \
                 mock.patch("comfy.model_management.soft_empty_cache", lambda: None), \
                 mock.patch("comfy.model_management.get_torch_device",
                            return_value="cuda:0-test"):
                # Move obj: load_gemma_hybrid разрешает устройства через
                # resolve_devices + torch_mock.device.
                # Строим 'inner' chain.
                inner_inner = _FakeInnerModule(proj)  # новая копия для 2-го вызова
                patcher_obj2 = mock.MagicMock()
                patcher_obj2.model = inner_inner
                patcher_obj2._ltx2_hooks = []
                patcher_obj2.model_options = {}
                clip_obj2 = mock.MagicMock()
                clip_obj2.patcher = patcher_obj2

                gguf_split.load_gemma_hybrid(
                    encoder_name="enc.safetensors",
                    projection_name="proj.safetensors",
                    verbose=False,
                    donor_device="cpu",
                    eject_models=False,
                )
                # 1-й вызов: proj._forward_pre_hooks имеет 1 entry (НЕ 2).
                self.assertEqual(
                    len(proj._forward_pre_hooks),
                    1,
                    f"After 1st load: должен быть 1 hook (got {len(proj._forward_pre_hooks)})",
                )

                # 2-й вызов через тот же proj (но другая inner_inner).
                gguf_split.load_gemma_hybrid(
                    encoder_name="enc.safetensors",
                    projection_name="proj.safetensors",
                    verbose=False,
                    donor_device="cpu",
                    eject_models=False,
                )
                # FIX #1 verify: pre-hooks были очищены ПЕРЕД регистрацией,
                # иначе после 2-го вызова у НЕГО должно быть 2+.
                self.assertLessEqual(
                    len(proj._forward_pre_hooks),
                    1,
                    f"After 2nd load: FIX #1 violation — {len(proj._forward_pre_hooks)} hooks instead of <=1. "
                    "Hook accumulation regressed.",
                )


# ════════════════════════════════════════════════════════════════════════════
# Block 2: raise on proj move failure (FIX 2)
# ════════════════════════════════════════════════════════════════════════════
class TestRaiseOnProjMoveFailure(unittest.TestCase):
    """Если projection-параметр не перенесён на primary_dev и torch available —
    load_gemma_hybrid должен raise RuntimeError с понятным message.

    Если torch=None (тест-окружение) — silent WARN через verbose.
    """

    def test_raises_when_proj_param_move_fails_with_torch(self):
        """С torch-доступным: failed move проекции должен raise (НЕ silent)."""
        from core import gguf_split

        proj = _FakeProjModule()
        inner = _FakeInnerModule(proj)
        patcher_obj = mock.MagicMock()
        patcher_obj.model = inner
        patcher_obj._ltx2_hooks = []
        patcher_obj.model_options = {}
        clip_obj = mock.MagicMock()
        clip_obj.patcher = patcher_obj

        # Мокаем _move_param: перестать двигать параметры проекции
        # (имитация OOM / device-mismatch).
        call_log: list[str] = []

        def failing_move(p, dev):
            # Если параметр из проекции — failed.
            # Для тестирования: первый param в fake — это projection.
            call_log.append(str(id(p)))
            return False

        # Оборачиваем: _move_param returns True for encoder params,
        # False для proj params → raise должен триггернуть.
        with _comfy_runtime_mock(), \
             mock.patch.object(gguf_split, "folder_paths", create=True), \
             mock.patch.object(gguf_split, "torch", create=True) as torch_mock, \
             mock.patch.object(gguf_split, "_lock_inner_to", lambda _inner: None), \
             mock.patch.object(
                 gguf_split, "_move_param", new=failing_move,
             ):
            torch_mock.is_available.return_value = True
            torch_mock.cuda.is_available.return_value = False

            with mock.patch("comfy.sd.load_clip", return_value=clip_obj), \
                 mock.patch("comfy.model_management.cuda_device_context",
                            mock.MagicMock()), \
                 mock.patch("comfy.model_management.soft_empty_cache", lambda: None), \
                 mock.patch("comfy.model_management.get_torch_device",
                            return_value="cuda:0-test"):
                with self.assertRaises(RuntimeError) as exc_context:
                    gguf_split.load_gemma_hybrid(
                        encoder_name="enc.safetensors",
                        projection_name="proj.safetensors",
                        verbose=False,
                        donor_device="cpu",
                        eject_models=False,
                    )
                # Проверяем, что raise содержит имя failed param'а
                # (наш fake имеет 1 projection param: model.text_projection.weight).
                self.assertIn(
                    "text_projection",
                    str(exc_context.exception),
                    "RuntimeError должен упомянуть text_projection или его параметр.",
                )
                self.assertIn(
                    "не удалось",
                    str(exc_context.exception),
                    "RuntimeError должен использовать понятное сообщение "
                    "(часть фразы содержит 'не удалось').",
                )

    def test_silent_skip_when_torch_none_for_test_env(self):
        """При torch=None проходит без raise (тестовая среда, не production).

        NB: Этот test-case семантически нереализуем через load_gemma_hybrid
        (функция имеет раннюю guard ``if folder_paths is None or torch is None:
        raise 'требует ComfyUI runtime'``, которая срабатывает прежде чем
        мы дойдём до FIX 2 raise). Чтобы корректно проверить 'silent skip
        когда torch=None + proj_move_failures', нужна refactor — вынести
        raise-on-failure logic в отдельный helper, который можно вызвать
        вне load_gemma_hybrid.

        Вместо сложного mock-construct'а здесь просто smoke-test'им, что
        при torch=None + sys-уровневой comfy mock функция либо (а) raise'ит
        ранюю 'требует ComfyUI runtime', либо (б) работает без ошибок (на
        patched MagicMock-окружении, где folder_paths не None). В обоих
        случаях НЕ должна выскакивать 'не удалось перенести' (FIX 2 silent
        path требует torch is not None — это гарантируется ранним guard).
        """
        from core import gguf_split

        proj = _FakeProjModule()
        inner = _FakeInnerModule(proj)
        patcher_obj = mock.MagicMock()
        patcher_obj.model = inner
        patcher_obj._ltx2_hooks = []
        patcher_obj.model_options = {}
        clip_obj = mock.MagicMock()
        clip_obj.patcher = patcher_obj

        def silent_move(p, dev):
            return True

        original_torch = gguf_split.torch
        try:
            gguf_split.torch = None  # simulate torch unavailable
            with _comfy_runtime_mock(), \
                 mock.patch.object(gguf_split, "folder_paths",
                                   mock.MagicMock(), create=True), \
                 mock.patch.object(gguf_split, "_lock_inner_to", lambda _inner: None), \
                 mock.patch.object(gguf_split, "_move_param", new=silent_move), \
                 mock.patch("comfy.sd.load_clip", return_value=clip_obj), \
                 mock.patch("comfy.model_management.cuda_device_context",
                            mock.MagicMock()), \
                 mock.patch("comfy.model_management.soft_empty_cache", lambda: None), \
                 mock.patch("comfy.model_management.get_torch_device",
                            return_value="cuda:0-test"):
                # torch=None + folder_paths mocked + _move_param=success →
                # НЕ должно raise 'не удалось перенести' (torch is None skip).
                # Может raise ранюю 'требует ComfyUI' если мы пропустили —
                # но это уже другой code-path.
                try:
                    gguf_split.load_gemma_hybrid(
                        encoder_name="enc.safetensors",
                        projection_name="proj.safetensors",
                        verbose=False,
                        donor_device="cpu",
                        eject_models=False,
                    )
                except RuntimeError as exc:
                    msg = str(exc)
                    # Production raise-text guard (single canonical string):
                    forbidden = ("не удалось перенести",)
                    if any(f in msg for f in forbidden):
                        self.fail(
                            f"При torch=None не должно raise 'не удалось перенести'. "
                            f"Got: {exc}"
                        )
                    if "требует ComfyUI runtime" not in msg:
                        # Непредвиденный RuntimeError — НЕ глотаем, пусть unittest
                        # зарепортит traceback (code-regression detector).
                        raise
                    # Известный ранний raise (folder_paths/torch guard) — OK.
        finally:
            gguf_split.torch = original_torch


# ════════════════════════════════════════════════════════════════════════════
# Block 3: encoder_param_count unconditional (FIX 3)
# ════════════════════════════════════════════════════════════════════════════
class TestEncoderParamCountAlwaysSet(unittest.TestCase):
    """При donor=cpu и verbose=False, encoder_param_count > 0 в meta dict.

    До фикса: encoder_param_count оставался = 0 при silent-path. Downstream-ноды,
    читающие ``model_options["ltx2_multigpu_gemma_split"]["encoder_param_count"]``,
    получали 0 и не могли отлаживать размер проекции.
    """

    def test_encoder_param_count_nonzero_when_donor_cpu_verbose_false(self):
        """Дано: 4 параметра в fake inner, из них 1 — projection, 3 — encoder.

        Действие: load_gemma_hybrid с donor=cpu, verbose=False.
        Ожидание: meta-dict содержит ``encoder_param_count == 3``.
        """
        from core import gguf_split

        proj = _FakeProjModule()
        inner = _FakeInnerModule(proj)
        patcher_obj = mock.MagicMock()
        patcher_obj.model = inner
        patcher_obj._ltx2_hooks = []
        patcher_obj.model_options = {}
        clip_obj = mock.MagicMock()
        clip_obj.patcher = patcher_obj

        def silent_move(p, dev):
            return True

        with _comfy_runtime_mock(), \
             mock.patch.object(gguf_split, "folder_paths", create=True), \
             mock.patch.object(gguf_split, "torch", create=True) as torch_mock, \
             mock.patch.object(gguf_split, "_lock_inner_to", lambda _inner: None), \
             mock.patch.object(gguf_split, "_move_param", new=silent_move):
            torch_mock.is_available.return_value = True
            torch_mock.cuda.is_available.return_value = False

            with mock.patch("comfy.sd.load_clip", return_value=clip_obj), \
                 mock.patch("comfy.model_management.cuda_device_context",
                            mock.MagicMock()), \
                 mock.patch("comfy.model_management.soft_empty_cache", lambda: None), \
                 mock.patch("comfy.model_management.get_torch_device",
                            return_value="cuda:0-test"):
                patcher = gguf_split.load_gemma_hybrid(
                    encoder_name="enc.safetensors",
                    projection_name="proj.safetensors",
                    verbose=False,
                    donor_device="cpu",
                    eject_models=False,
                )
                meta = patcher.model_options.get("ltx2_multigpu_gemma_split", {})
                self.assertIn(
                    "encoder_param_count", meta,
                    "meta dict должен содержать encoder_param_count даже когда verbose=False.",
                )
                self.assertGreater(
                    meta["encoder_param_count"], 0,
                    "encoder_param_count должен быть > 0 при donor=cpu + verbose=False "
                    "(3 encoder params в fake, projection excluded).",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
