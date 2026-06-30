"""Custom nodes for ComfyUI-LTX2-MultiGPU (Author: THE-ANGEL-AI).

Правила реализации (из D:\\ComfyUI\\Docs\\docs\\00-AGENT-RULES.md):
  R1 — проверять реальный API в D:\\ComfyUI\\comfy\\{model_management,model_patcher,sd}.py
  R2 — V1 API (NODE_CLASS_MAPPINGS / INPUT_TYPES) — НЕ мешать с V3
  R3 — НЕ хардкодить cuda:0/cuda:1; тянуть устройство из mm.get_torch_device()
  R4 — типы строго MODEL / CLIP / VAE / LATENT (FUNCTION возвращает tuple, длина ⇔ RETURN_TYPES)
  R5/R6 — уникальные префиксы в ключах (LTX2_MultiGPU_...), тяжёлые импорты в try/except
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]

try:
    import folder_paths  # предоставляется ComfyUI при запуске
    _FOLDER_PATHS_OK = True
except Exception:  # noqa: BLE001
    folder_paths = None  # type: ignore[assignment]
    _FOLDER_PATHS_OK = False


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helpers (FIX LOW #4: дедупликация _donor_choices)
# ─────────────────────────────────────────────────────────────────────────────
def _cuda_donor_choices(include_cpu: bool = False) -> list[str]:
    """FIX LOW #4+#6+#LOW_cpu_machine: динамический donor_device-список.

    Использует torch.cuda.device_count() если torch доступен И CUDA активен;
    на CPU-only машинах / torch=None fallback не показывает cuda:* опции
    (юзер не должен видеть то, что упадёт с RuntimeError на load).

    Args:
        include_cpu: True для Gemma (encoder допускает CPU — он не
                      участвует в sampling-loop); False для DiT (DiT
                      не может жить на CPU во время KSampler-step).
    """
    choices: list[str] = ["auto"]
    if torch is not None and torch.cuda.is_available():
        try:
            n = int(torch.cuda.device_count())
            for i in range(n):
                choices.append(f"cuda:{i}")
        except (RuntimeError, AssertionError):
            pass
    # Базовые cuda:0/cuda:1 присутствуют ВСЕГДА — это не зависит от torch
    # (сохраняет контракт исходного _DONOR_DEVICE_CHOICES_DIT).
    for tag in ("cuda:0", "cuda:1"):
        if tag not in choices:
            choices.append(tag)
    if include_cpu:
        choices.append("cpu")
    return choices


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 1. LTX2_MultiGPU_HybridSplitLoader
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_HybridSplitLoader:
    """Заменяет UnetLoaderGGUFDisTorch2MultiGPU для LTX 2.3 GGUF.

    Возвращает (R3) ModelPatcher-совместимый объект — совместим с LoRA,
    offload, sampler. Реализация сплита лежит в core/gguf_split.py.

    UI mirror GemmaHybrid: добавлен `donor_device` (auto/cuda:0/cuda:1).
    NB: `eject_models` НЕ добавляется — DiT вызывается каждый sampling-step,
    offload DiT → CPU = PCIe death и sampling stall. cpу как donor_device
    отвергается уже в INPUT_TYPES (выбор ограничен 3-мя опциями).
    """

    NODE_ID = "LTX2_MultiGPU_HybridSplitLoader"
    # Russian display name for users (grouped by CATEGORY).
    # NODE_ID (technical class key) preserved for workflow_api compat.
    DISPLAY_NAME = "🔀 Разделитель DiT (2×GPU)"

    FUNCTION = "load"
    CATEGORY = "THE-ANGEL-AI"
    OUTPUT_NODE = False

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    # DiT-specific donor_device: НЕТ 'cpu' — DiT не может быть на CPU во время sampling.
    # Решение принято в design-discussion: см. commit message + docstring class.

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        # FIX dropdown-bug: ВСЕГДА используем folder_paths.get_filename_list()
        # если _FOLDER_PATHS_OK — даже при пустом списке ComfyUI рендерит
        # выпадающий список (пустой dropdown ЛУЧШЕ чем bare text input).
        # "(choices_tuple, )" = ComfyUI-формат для dropdown-виджета.
        if _FOLDER_PATHS_OK:
            try:
                unet_choices: list = folder_paths.get_filename_list("diffusion_models")  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                unet_choices = []
            unet_opts: tuple = (unet_choices,)
        else:
            unet_opts = ("STRING", {"default": ""})
        return {
            "required": {
                "unet_name": unet_opts,
                "split_strategy": (
                    ["blocks_50_50", "blocks_30_70", "pipeline", "single_cuda0", "single_cuda1"],
                    {"default": "blocks_50_50"},
                ),
                # FIX LOW #4+#6: module-level dynamic donor_device.
                "donor_device": (_cuda_donor_choices(include_cpu=False), {"default": "auto"}),
                "verbose_log": ("BOOLEAN", {"default": False}),
            }
        }

    def load(
        self,
        unet_name: str,
        split_strategy: str,
        donor_device: str,
        verbose_log: bool,
    ) -> tuple:
        """Делегирует в core.gguf_split.hybrid_split_gguf (R4: tuple-возврат)."""
        if folder_paths is None:
            raise RuntimeError(
                "folder_paths недоступен — узел должен запускаться внутри ComfyUI"
            )

        from core.gguf_split import hybrid_split_gguf

        try:
            model_patcher = hybrid_split_gguf(
                gguf_name=unet_name,
                strategy=split_strategy,
                verbose=verbose_log,
                donor_device=donor_device,
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                f"ComfyUI-LTX2-MultiGPU: {exc}"
            ) from exc

        if verbose_log:
            print(
                f"[ComfyUI-LTX2-MultiGPU] Loaded {unet_name} with strategy "
                f"{split_strategy}, donor={donor_device}"
            )

        return (model_patcher,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 2. LTX2_MultiGPU_GemmaHybridLoader
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_GemmaHybridLoader:
    """Жёсткая загрузка Gemma 3 12B FP4: encoder → donor_device, projection → cuda:0.

    UI совместим с DualCLIPLoaderDisTorch2MultiGPU (см. скриншот пользователя):
      donor_device   куда грузить encoder (auto / cuda:0 / cuda:1 / cpu).
                     auto ⇒ mm-derived secondary (cuda:1 в dual-GPU config).
      eject_models   True ⇒ после load: offload_device=CPU + soft_empty_cache.
                     NB: под Risk #7 lock (.to() no-op) sampler не сможет
                     re-load projection обратно на cuda:0 после eject —
                         используйте только если уверены.
      verbose_log    печатать per-component allocation log (encoder/proj GB,
                     VRAM free до/после).
    """

    NODE_ID = "LTX2_MultiGPU_GemmaHybridLoader"
    # Russian display name for users (grouped by CATEGORY).
    # NODE_ID (technical class key) preserved for workflow_api compat.
    DISPLAY_NAME = "📝 Dual CLIP Загрузчик (Gemma 3)"

    FUNCTION = "load"
    CATEGORY = "THE-ANGEL-AI"
    OUTPUT_NODE = False

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)

    # Donor-device options FIX LOW #4: дедуп через module-level helper
    # ``_cuda_donor_choices(include_cpu=True)``.

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        # FIX dropdown-bug: ВСЕГДА используем folder_paths.get_filename_list()
        # если _FOLDER_PATHS_OK — даже при пустом списке ComfyUI рендерит
        # выпадающий список (пустой dropdown ЛУЧШЕ чем bare text input).
        # Gemma-файлы (.safetensors) могут лежать в text_encoders/ ИЛИ clip/ —
        # объединяем обе папки для encoder и projection.
        if _FOLDER_PATHS_OK:
            try:
                enc_folder = folder_paths.get_filename_list("text_encoders")  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                enc_folder = []
            try:
                clip_folder = folder_paths.get_filename_list("clip")  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                clip_folder = []
            # Оба виджета видят файлы из обеих папок (как старый код через
            # ``for folder in (...): if items: break``, но БЕЗ guard'а).
            merged: list = list(dict.fromkeys(enc_folder + clip_folder))  # dedup
            enc_opts: tuple = (merged,)
            proj_opts: tuple = (merged,)
        else:
            enc_opts = ("STRING", {"default": ""})
            proj_opts = ("STRING", {"default": ""})
        return {
            "required": {
                "clip_name1": enc_opts,
                "projection_name": proj_opts,
                "donor_device": (_cuda_donor_choices(include_cpu=True), {"default": "auto"}),
                "eject_models": ("BOOLEAN", {"default": False}),
                "verbose_log": ("BOOLEAN", {"default": False}),
            }
        }

    def load(
        self,
        clip_name1: str,
        projection_name: str,
        donor_device: str,
        eject_models: bool,
        verbose_log: bool,
    ) -> tuple:
        from core.gguf_split import load_gemma_hybrid

        # load_gemma_hybrid raises RuntimeError с понятными причинами
        # (усл.). Не мапим в RuntimeError повторно — propagate as is.
        clip = load_gemma_hybrid(
            encoder_name=clip_name1,
            projection_name=projection_name,
            verbose=verbose_log,
            donor_device=donor_device,
            eject_models=eject_models,
        )

        return (clip,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 3. LTX2_MultiGPU_MemoryDiagnostics
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_MemoryDiagnostics:
    """Pre-flight VRAM checker: dry-load прогон + nvidia-smi лог в консоль."""

    NODE_ID = "LTX2_MultiGPU_MemoryDiagnostics"
    # Russian display name for users (grouped by CATEGORY).
    # NODE_ID (technical class key) preserved for workflow_api compat.
    DISPLAY_NAME = "🩺 Диагностика VRAM"

    FUNCTION = "diagnose"
    CATEGORY = "THE-ANGEL-AI"
    OUTPUT_NODE = True  # для ui={'report': [report]}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        # FIX dropdown-bug: MemoryDiagnostics тоже должен показывать
        # выпадающие списки файлов, а не голые текстовые поля.
        # Gemma (.safetensors) может быть в text_encoders/ ИЛИ clip/.
        if _FOLDER_PATHS_OK:
            try:
                unet_choices: list = folder_paths.get_filename_list("diffusion_models")  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                unet_choices = []
            try:
                gemma_enc = folder_paths.get_filename_list("text_encoders")  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                gemma_enc = []
            try:
                gemma_clip = folder_paths.get_filename_list("clip")  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                gemma_clip = []
            gemma_choices: list = list(dict.fromkeys(gemma_enc + gemma_clip))  # dedup
            unet_opts: tuple = (unet_choices,)
            gemma_opts: tuple = (gemma_choices,)
        else:
            unet_opts = ("STRING", {"default": ""})
            gemma_opts = ("STRING", {"default": ""})
        return {
            "required": {
                "unet_name": unet_opts,
                "gemma_name": gemma_opts,
                "purge_cache": ("BOOLEAN", {"default": True}),
            }
        }

    def diagnose(
        self,
        unet_name: str,
        gemma_name: str,
        purge_cache: bool,
    ) -> tuple:
        """R4: V1 tuple-контракт — возвращаем ровно ``(report,)``.

        ComfyUI ``execution.py`` анпакает выходы по схеме
        ``val, = node.FUNCTION(...)``. Dict-возврат формата
        ``{"ui": ..., "result": (...)}`` понимают только новые версии ComfyUI
        (V3 preview API); на старых / Kaggle-зерокопии — крашится с
        ``ValueError: too many values to unpack (expected 1)``.
        Для консольного preview достаточно print() — он попадает в ComfyUI log
        stdout и виден без UI.
        """
        from core.memory_tracker import estimate_vram_budget

        report = estimate_vram_budget(
            gguf_name=unet_name,
            gemma_name=gemma_name,
            purge_cache=purge_cache,
        )
        print(f"[ComfyUI-LTX2-MultiGPU]\n{report}")
        return (report,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 4. LTX2_MultiGPU_DeviceStrategy
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_DeviceStrategy:
    """Переключатель глобальной стратегии распределения через mm."""

    NODE_ID = "LTX2_MultiGPU_DeviceStrategy"
    # Russian display name for users (grouped by CATEGORY="THE-ANGEL-AI").
    # CATEGORY prefix adds the brand tag automatically in ComfyUI's Add Node menu,
    # so we don't repeat it here. NODE_ID (technical class key) preserved.
    DISPLAY_NAME = "⚙️ Стратегия GPU (hot-switch)"

    FUNCTION = "apply_strategy"
    CATEGORY = "THE-ANGEL-AI"
    OUTPUT_NODE = False

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "strategy": (
                    ["blocks_50_50", "blocks_30_70", "pipeline", "single_cuda0", "single_cuda1"],
                    {"default": "blocks_50_50"},
                ),
                # NEW (v0.2.1): donor_device widget — раньше нода захардкодила
                # ``secondary_dev`` из ``resolve_devices()``, из-за чего смена
                # стратегии на лету ИГНОРИРОВАЛА user override из HybridSplitLoader.
                # Теперь оба loadера (нода + HybridSplitLoader) могут горячо менять
                # стратегию с учётом оригинального donor_device.
                "donor_device": (_cuda_donor_choices(include_cpu=False), {"default": "auto"}),
            },
            "optional": {
                # FIX MEDIUM_apply_strategy: forward verbose из UI-виджета в
                # core.apply_strategy(verbose=verbose_log). WARN уже unconditional
                # в core (cab22dc), verbose используется для детального per-block лога.
                "verbose_log": ("BOOLEAN", {"default": False}),
            },
        }

    def apply_strategy(
        self,
        model,
        strategy: str,
        donor_device: str = "auto",
        verbose_log: bool = False,
    ) -> tuple:
        from core.gguf_split import apply_strategy as _apply

        try:
            new_patcher = _apply(
                patcher=model,
                strategy=strategy,
                verbose=verbose_log,
                donor_device=donor_device,
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                f"ComfyUI-LTX2-MultiGPU: {exc}"
            ) from exc
        return (new_patcher,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 5. LTX2_MultiGPU_VRAMParking
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_VRAMParking:
    """Временно убирает DiT из VRAM для освобождения памяти под VAE/upscale.

    Между Pass 1 (KSampler) и Pass 2 (KSampler) ComfyUI делает:
      VAE decode → upscale → VAE encode.
    DiT блоки (~9 GB на каждой карте) конкурируют с VAE/upscaler за VRAM.

    Парковка переносит ВСЕ DiT блоки + служебные слои на CPU, освобождая
    VRAM под VAE-стадии. После VAE encode — распарковка возвращает блоки
    на исходные GPU в точности с исходной стратегией.
    """

    NODE_ID = "LTX2_MultiGPU_VRAMParking"
    DISPLAY_NAME = "🅿️ Парковка DiT (VRAM↔CPU)"

    FUNCTION = "apply"
    CATEGORY = "THE-ANGEL-AI"
    OUTPUT_NODE = False

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "park_in_cpu": ("BOOLEAN", {"default": True,
                    "label_on": "Запарковать в CPU",
                    "label_off": "Вернуть на GPU"}),
                "verbose_log": ("BOOLEAN", {"default": False}),
            }
        }

    def apply(
        self,
        model,
        park_in_cpu: bool,
        verbose_log: bool = False,
    ) -> tuple:
        from core.vram_parking import park_dit, unpark_dit

        if park_in_cpu:
            park_dit(model)
        else:
            unpark_dit(model)

        if verbose_log:
            state = "запаркован" if park_in_cpu else "распаркован"
            print(
                f"[ComfyUI-LTX2-MultiGPU] VRAMParking: DiT {state} "
                f"({'park_in_cpu=True' if park_in_cpu else 'park_in_cpu=False'})"
            )

        return (model,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 6. LTX2_MultiGPU_SageAttention
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_SageAttention:
    """Ускоритель внимания через SageAttention-SM75 (оптимизация под T4/SM75).

    Заменяет стандартное scaled_dot_product_attention на квантованное
    внимание (INT8 QK^T + FP16 PV) через SageAttention-SM75 — форк,
    адаптированный для Turing SM75 (T4) через Triton fallback.

    Ожидаемое ускорение: ~1.5x end-to-end (внимание = 25-35% DiT compute).

    Автоопределение: если sageattn НЕ установлен — тихо возвращает
    модель без патча (стандартное внимание).
    """

    NODE_ID = "LTX2_MultiGPU_SageAttention"
    DISPLAY_NAME = "⚡ SageAttention (T4 турбо)"

    FUNCTION = "apply"
    CATEGORY = "THE-ANGEL-AI"
    OUTPUT_NODE = False

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True,
                    "label_on": "Включить SageAttn",
                    "label_off": "Выключить"}),
                "verbose_log": ("BOOLEAN", {"default": False}),
            }
        }

    def apply(
        self,
        model,
        enable: bool,
        verbose_log: bool = False,
    ) -> tuple:
        from core.sage_attention import get_sageattn_patch

        # Клонируем patcher чтобы не мутировать оригинальный граф
        # (shallow copy model_options).
        patcher = model.clone()

        if enable:
            patch = get_sageattn_patch(verbose=verbose_log)
            if patch:
                opts = patcher.model_options.setdefault("attention_patch", {})
                opts.update(patch)
                if verbose_log:
                    print(
                        "[ComfyUI-LTX2-MultiGPU] SageAttention патч "
                        "установлен в model_options."
                    )
            elif verbose_log:
                print(
                    "[ComfyUI-LTX2-MultiGPU] SageAttention НЕ активирован "
                    "(модуль sageattn не найден)."
                )
        else:
            # Очищаем патч если был установлен ранее в цепочке нод.
            if "attention_patch" in patcher.model_options:
                patcher.model_options["attention_patch"].pop("default", None)
                # Убираем пустой attention_patch ключ (чистота model_options).
                if not patcher.model_options["attention_patch"]:
                    del patcher.model_options["attention_patch"]
            if verbose_log:
                print("[ComfyUI-LTX2-MultiGPU] SageAttention отключён.")

        return (patcher,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 7. LTX2_MultiGPU_VAELoader
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_VAELoader:
    """VAE загрузчик с выбором GPU для VAE Decode/Encode.

    На T4×2 (14.5 GB каждая) VAE decode (~3-5 GB VRAM) конкурирует
    с DiT/Gemma за память. Эта нода позволяет явно выбрать GPU для
    VAE, чтобы избежать OOM между Pass 1 и Pass 2 (VAE decode → upscale
    → VAE encode).

    Использует ``comfy.sd.load_vae()`` для загрузки и ``.to(device)``
    для явного размещения на выбранном GPU.
    """

    NODE_ID = "LTX2_MultiGPU_VAELoader"
    DISPLAY_NAME = "🖼️ VAE Загрузчик (GPU)"

    FUNCTION = "load"
    CATEGORY = "THE-ANGEL-AI"
    OUTPUT_NODE = False

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        # FIX dropdown-bug: всегда dropdown (даже пустой) если _FOLDER_PATHS_OK.
        if _FOLDER_PATHS_OK:
            try:
                vae_choices: list = folder_paths.get_filename_list("vae")  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                vae_choices = []
            vae_opts: tuple = (vae_choices,)
        else:
            vae_opts = ("STRING", {"default": ""})
        return {
            "required": {
                "vae_name": vae_opts,
                # VAE может жить на CPU между вызовами — include_cpu=True.
                "donor_device": (_cuda_donor_choices(include_cpu=True), {"default": "auto"}),
                "verbose_log": ("BOOLEAN", {"default": False}),
            }
        }

    def load(
        self,
        vae_name: str,
        donor_device: str,
        verbose_log: bool,
    ) -> tuple:
        """Загружает VAE и размещает на выбранном GPU."""
        if folder_paths is None or torch is None:
            raise RuntimeError(
                "LTX2_MultiGPU_VAELoader требует ComfyUI runtime (folder_paths, torch)"
            )

        try:
            from comfy import sd as comfy_sd  # type: ignore[import-not-found]
            from comfy import model_management as mm  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"comfy.sd / model_management недоступны: {exc}"
            ) from exc

        # ── Разрешаем device ────────────────────────────────────────────
        primary_dev = mm.get_torch_device()

        # Определяем secondary (cuda:1 если есть, иначе primary)
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            idx = primary_dev.index if primary_dev.index is not None else 0
            secondary_dev = torch.device("cuda", int((idx + 1) % torch.cuda.device_count()))
        else:
            secondary_dev = primary_dev

        spec = (donor_device or "auto").strip().lower()
        if spec == "auto":
            target_dev = secondary_dev
        elif spec == "cuda:0":
            target_dev = primary_dev
        elif spec == "cuda:1":
            target_dev = secondary_dev
        elif spec == "cpu":
            target_dev = torch.device("cpu")
        else:
            try:
                target_dev = torch.device(spec)
            except Exception:  # noqa: BLE001
                target_dev = secondary_dev

        # ── Загружаем VAE ───────────────────────────────────────────────
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if not vae_path:
            raise FileNotFoundError(
                f"VAE '{vae_name}' не найден в vae/"
            )

        try:
            vae = comfy_sd.load_vae(vae_path)
        except Exception as exc:
            raise RuntimeError(
                f"comfy.sd.load_vae failed для {vae_name!r}: {exc}"
            ) from exc

        if vae is None:
            raise RuntimeError(
                f"comfy.sd.load_vae вернул None для {vae_name!r}"
            )

        # ── Размещаем first_stage_model на целевом устройстве ───────────
        # VAE объект в ComfyUI содержит .first_stage_model (nn.Module).
        # Перемещаем его на target_dev для VAE Decode/Encode.
        if hasattr(vae, "first_stage_model"):
            try:
                vae.first_stage_model.to(target_dev, non_blocking=False)
            except Exception as exc:  # noqa: BLE001
                if verbose_log:
                    print(
                        f"[ComfyUI-LTX2-MultiGPU] WARN: VAE first_stage_model.to({target_dev}) "
                        f"failed: {exc}"
                    )
        elif hasattr(vae, "to"):
            try:
                vae.to(target_dev)
            except Exception as exc:  # noqa: BLE001
                if verbose_log:
                    print(
                        f"[ComfyUI-LTX2-MultiGPU] WARN: vae.to({target_dev}) failed: {exc}"
                    )

        if verbose_log:
            try:
                import os
                vae_gb = os.path.getsize(vae_path) / (1024 ** 3)
            except Exception:  # noqa: BLE001
                vae_gb = 0.0
            print(
                f"[ComfyUI-LTX2-MultiGPU] VAE loaded: {vae_name} "
                f"({vae_gb:.2f} GB) @ {target_dev} (donor={donor_device!r})"
            )

        return (vae,)


__all__ = [
    "LTX2_MultiGPU_HybridSplitLoader",
    "LTX2_MultiGPU_GemmaHybridLoader",
    "LTX2_MultiGPU_MemoryDiagnostics",
    "LTX2_MultiGPU_DeviceStrategy",
    "LTX2_MultiGPU_VRAMParking",
    "LTX2_MultiGPU_SageAttention",
    "LTX2_MultiGPU_VAELoader",
]
