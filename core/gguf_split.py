"""core/gguf_split.py — ядро Hybrid Split для LTX 2.3 DiT.

⚠️ Phase 2 реализация. Использует:
  - `comfy.model_management` (mm) для device-resolution (R3)
  - `comfy.model_patcher.ModelPatcher` для обёртки (R3)
  - city96 `UnetLoaderGGUF.load_unet` для GGUF dequant (lazy GGMLOps)
  - srijithr-паттерн: register_forward_hook с передачей hidden_states.to(target_device)

НЕ хардкодит cuda:0/cuda:1, НЕ возвращает голый nn.Module.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]

try:
    import folder_paths  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    folder_paths = None  # type: ignore[assignment]


# Кол-во transformer blocks в DiT LTX 2.3 (см. MODEL_FACTS §3)
LTX2_DIT_BLOCK_COUNT = 44

# Стратегии split (согласовано с MODEL_FACTS §6)
STRATEGIES = (
    "blocks_50_50",    # default: DiT 0–21 -> cuda:0, 22–43 -> cuda:1
    "blocks_30_70",    # PCIe friendly: 0–13 -> cuda:0, 14–43 -> cuda:1
    "pipeline",        # fallback C: DiT целиком на cuda:1, Gemma на cuda:0
    "single_cuda0",    # debug: всё на cuda:0
    "single_cuda1",    # debug: всё на cuda:1
)

# Префиксы тензоров DiT (см. MODEL_FACTS §3). Source-of-truth для classify_tensors.
DIFFUSION_BLOCK_PREFIX = "model.diffusion_model.transformer_blocks."
DIFFUSION_EMBED_PREFIXES = (
    "model.diffusion_model.time_embed.",
    "model.diffusion_model.adaln",
    "model.diffusion_model.patchify_proj.",
    "model.diffusion_model.proj_in.",
    "model.diffusion_model.norm_in.",
)


__all__ = [
    "LTX2_DIT_BLOCK_COUNT",
    "STRATEGIES",
    "classify_tensor",
    "resolve_devices",
    "resolve_donor_device",
    "hybrid_split_gguf",
    "load_gemma_hybrid",
    "apply_strategy",
]


def classify_tensor(name: str) -> str:
    """Классифицирует тензор DiT для распределения по GPU.

    Возвращает одно из:
      - "block"   → tensor_blocks.{N}.* (DiT transformer block)
      - "embed"   → embedding / AdaLN / outer (должен быть на cuda:0)
      - "head"    → proj_out / norm_out
      - "other"   → всё остальное (VAE / text encoder / внешнее)
    """
    if name.startswith(DIFFUSION_BLOCK_PREFIX):
        return "block"
    for pfx in DIFFUSION_EMBED_PREFIXES:
        if name.startswith(pfx):
            return "embed"
    if name.startswith("model.diffusion_model.proj_out.") or name.startswith(
        "model.diffusion_model.norm_out."
    ):
        return "head"
    if name.startswith("model.diffusion_model."):
        return "other_diffusion"
    return "other"


def _mm_primary() -> Any:
    """Возвращает primary torch.device через mm.get_torch_device()."""
    try:
        from comfy import model_management as mm  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "comfy.model_management недоступен — запустите пакет внутри ComfyUI"
        ) from exc
    return mm.get_torch_device()


def _mm_secondary(primary: Any) -> Any:
    """Возвращает second GPU-устройство без хардкода cuda:0/cuda:1."""
    if torch is None or not torch.cuda.is_available():
        return primary
    if primary.type != "cuda":
        return primary
    n = int(torch.cuda.device_count())
    if n < 2:
        return primary
    # primary.index может быть None (вызов `torch.device('cuda')` без индекса)
    idx = primary.index if primary.index is not None else 0
    return torch.device("cuda", int((idx + 1) % n))


def resolve_devices() -> tuple[Any, Any]:
    """R3-совместимый resolve: primary + secondary.

    Возвращает (cuda0_like, cuda1_like). Если есть только одна карта, оба == primary.
    """
    primary = _mm_primary()
    secondary = _mm_secondary(primary)
    return primary, secondary


def resolve_donor_device(spec: str, primary: Any, secondary: Any) -> Any:
    """Приводит donor_device string → torch.device для encoder'а Gemma.

    Семантика:
      "auto"   → secondary (текущая поведение без изменений, по умолчанию cuda:1)
      "cuda:0" → primary   (override: encoder на primary, projection должна fit)
      "cuda:1" → secondary (явный secondary для single-GPU config неоднозначный,
                            но мы возвращаем secondary как «second available GPU»)
      "cpu"    → CPU (encoder не грузится на GPU — text_projection всё равно
                     поднимается на primary для sampling'а)

    Не матчит на «current torch device» по дизайну — это UI-driven выбор,
    который надо детерминированно меппить в устройства.
    """
    s = (spec or "auto").strip().lower()
    if s == "auto":
        return secondary
    if s == "cuda:0":
        return primary
    if s == "cuda:1":
        return secondary
    if s == "cpu":
        return torch.device("cpu") if torch is not None else "cpu"
    # Fallback для любого другого значения: парсим torch.device,
    # либо fallback на secondary без exception (вызывающий код ожидает device-like).
    try:
        if torch is not None:
            return torch.device(s)
    except Exception:  # noqa: BLE001
        pass
    return secondary


def _get_city96_patcher_class():
    """Ищет city96 UnetLoaderGGUF через registry ComfyUI.

    Используем глобальный :data:`comfy.nodes.NODE_CLASS_MAPPINGS` —
    туда ComfyUI собирает ВСЕ зарегистрированные custom node классы
    при загрузке. Это устраняет баг с raw ``__import__("nodes")``, который
    хватал либо наш собственный ``nodes.py``, либо ComfyUI root без
    прямого атрибута ``UnetLoaderGGUF``.

    Поддерживаем оба ID: ``UnetLoaderGGUF`` (текущий city96 main) и
    ``UNETLoaderGGUF`` (legacy API).
    """
    try:
        import nodes as comfy_root  # ComfyUI root: D:\\ComfyUI\\nodes.py
    except Exception as exc:
        raise RuntimeError(
            f"comfy root 'nodes' модуль недоступен — запуск вне ComfyUI: {exc}"
        ) from exc

    mappings = getattr(comfy_root, "NODE_CLASS_MAPPINGS", None) or {}
    if not mappings:
        raise RuntimeError(
            "NODE_CLASS_MAPPINGS пуст — ComfyUI ещё не зарегистрировал "
            "custom_nodes. Убедись, что ComfyUI-GGUF склонирован и его "
            "__init__.py лежит в custom_nodes/."
        )

    for class_id in ("UnetLoaderGGUF", "UNETLoaderGGUF"):
        cls = mappings.get(class_id)
        if cls is not None:
            return cls

    raise RuntimeError(
        "city96 ComfyUI-GGUF не найден: ни 'UnetLoaderGGUF', ни "
        "'UNETLoaderGGUF' не зарегистрированы в NODE_CLASS_MAPPINGS. "
        "Клонируй https://github.com/city96/ComfyUI-GGUF в custom_nodes/ "
        "(latest, не legacy)."
    )


def _call_with_ctx(device: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Вызывает fn под cuda_device_context(device) — безопасно переключает текущую CUDA."""
    try:
        from comfy import model_management as mm  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return fn(*args, **kwargs)
    cm = getattr(mm, "cuda_device_context", None)
    if cm is None:
        return fn(*args, **kwargs)
    with cm(device):
        return fn(*args, **kwargs)


def _install_cross_device_hook(module: Any, src_device: Any, dst_device: Any) -> Any:
    """forward_pre_hook srijithr-паттерна: перенос hidden_states между GPU.

    Регистрирует pre-hook на `module` (обычно `transformer_blocks[split_idx]` —
    первый блок на cuda:1). При вызове forward на этом блоке мы ДО его исполнения
    ДВИГАЕМ входной hidden_states с cuda:0 на cuda:1.

    Это лучше чем forward_post_hook на предыдущем блоке потому что:
      - не возвращает contract-modified tuple (нет gradient graph break в autograd);
      - естественно покрывает любую сигнатуру входа (tuple/list);
      - PyTorch позволяет; не мешает `register_forward_hook` для sampler.
    Возвращает handle для отмены.
    """
    if torch is None:
        return None
    # Degenerative guard (FIX MEDIUM #4 здесь): если src==dst, hook не нужен —
    # лишний overhead и не нужный autograd quirk.
    try:
        if torch.device(str(src_device)) == torch.device(str(dst_device)):
            return None
    except Exception:  # noqa: BLE001
        pass

    def _move(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            if obj.device == dst_device:
                return obj
            try:
                return obj.to(dst_device, non_blocking=True)
            except Exception:  # noqa: BLE001
                return obj.to(dst_device)
        if isinstance(obj, (tuple, list)):
            moved = [_move(x) for x in obj]
            return type(obj)(moved)
        # FIX CRITICAL #1: kwargs и прочие вложенные структуры.
        if isinstance(obj, dict):
            return {k: _move(v) for k, v in obj.items()}
        if isinstance(obj, set):
            return {_move(x) for x in obj}
        return obj

    def _pre_hook(_mod: Any, args: Any, kwargs: Any = None) -> Any:
        """FIX CRITICAL #1 + MEDIUM #2: ВСЕГДА возвращаем tuple ``(args, kwargs)``.

        PyTorch 2.0+ ``with_kwargs=True`` ожидает КОНТРАКТ (module, args, kwargs)
        → даже если kwargs пустой, возвращаем (args, {}) — иначе PyTorch
        может дропнуть kwargs в edge-case.

        Для 1.x fallback hook видит только args — legacy_branch возвращает
        lambda без kwargs и не вызывается с with_kwargs=True.
        """
        moved_args = _move(args)
        moved_kwargs = _move(kwargs) if kwargs is not None else {}
        return moved_args, moved_kwargs

    try:
        handle = module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    except TypeError:
        # Fallback для PyTorch < 2.0 — kwargs игнорируются.
        handle = module.register_forward_pre_hook(lambda _m, a: _move(a))
    return handle


def _lock_inner_to(inner: Any) -> None:
    """Risk #7 fix: monkey-patch ``inner.to(device)`` в no-op.

    ComfyUI sampler в начале каждой KSampler-step вызывает
    ``comfy.model_management.load_models_gpu([patcher])``, который
    BLANKET-вызывает ``inner.to(patcher.load_device)`` без проверки
    текущего device каждого параметра. После нашего ручного split
    (blocks 22..43 на cuda:1) это всё равно перетащит split-блоки
    обратно на ``load_device`` (= primary cuda:0) → OOM на 720p.

    Делаем ``inner.to`` no-op'ом. Patch идемпотентный через marker
    ``inner._ltx2_to_locked``. Оригинал сохраняем в
    ``inner._ltx2_original_to`` для rollback в тестах / debug.

    NB: лочим ТОЛЬКО bound ``inner.to`` (top-level module). Подмодули
    (``inner.submodule.to``) сохраняют nn.Module class method — это OK,
    sampler обычно зовёт ровно ``inner.to``.
    """
    try:
        if getattr(inner, "_ltx2_to_locked", False):
            return
        # Сохраняем настоящий to() ДО подмены.
        inner._ltx2_original_to = inner.to  # type: ignore[attr-defined]
        original_to = inner._ltx2_original_to

        def _is_device_arg(a: Any) -> bool:
            """Отдельный тue/false для одного аргумента: device-like или нет.

            Истино: ``torch.device('cuda:0')`` и строки ``'cuda'``/``'cuda:1'``/
            ``'cpu'``/``'mps'``/``'xpu'``/``'hpu'`` (covers main device-strings).
            Ложь: dtype/числа/None/всё остальное.
            """
            if torch is None:
                return False
            if isinstance(a, torch.device):
                return True
            if isinstance(a, str) and any(
                a.startswith(p) for p in ("cuda", "cpu", "mps", "xpu", "hpu")
            ):
                return True
            return False

        def _is_device_move(args: tuple, kwargs: dict) -> bool:
            """FIX HIGH #2 + MEDIUM #1: device-перенос — torch.device или
            'cuda*'/'cpu'/'mps'/'xpu'/'hpu' в args/kwargs.

            Defensive: если ``torch is None`` (модуль импортирован вне ComfyUI),
            sampler всё равно без активен → пропускаем все вызовы в original_to.
            """
            if torch is None:
                return False
            for a in args:
                if _is_device_arg(a):
                    return True
            return "device" in kwargs

        def _strip_device(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
            """Убирает device-args/kwargs, оставляя dtype/memory_format/non_blocking.

            Используется в ``_no_op_to`` чтобы блокировать **только** device-перенос,
            но дать проходить сопутствующим модификациям (`dtype=`, `memory_format=`,
            `non_blocking=True`, и пр.). Иначе KSampler после первого же нашего
            split-блокирующего вызова получал бы «застрявшую» dtype-cast, а это
            ведёт к sampler-uncacheable weight bites (review HIGH_FINAL_2).

            NB: top-level only — НЕ рекурсивен по tuple/list/dict. Это согласованно
            с реальной сигнатурой ``nn.Module.to()``, которая принимает только
            scalar позиционные opts (device/dtype/non_blocking/memory_format).
            Если когда-то появится nested device-bearing tuple (e.g. (some_callable,
            ('cuda:0',))) — добавим рекурсию зеркалируя ``_move_inputs_to`` helper
            в core/gguf_split.py:hidden_states-transport. Сейчас не нужно.
            """
            cleaned_args = tuple(a for a in args if not _is_device_arg(a))
            cleaned_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
            return cleaned_args, cleaned_kwargs

        def _no_op_to(*args: Any, **kwargs: Any) -> Any:
            """FIX HIGH #2 (final): блокируем только device-переносы sampler'а,
            но пробрасываем dtype/memory_format/non_blocking через original_to.

            Бывшая ошибка (review HIGH_FINAL_1): если ComfyUI вызывал
            ``inner.to(device='cuda:0', dtype=torch.float16)``, текущая
            реализация просто возвращала ``inner`` — компиляция/optimization
            sampler'а проглатывалась. Теперь: device args отфильтровываются,
            всё остальное forward'ится в ``original_to``.
            """
            if not _is_device_move(args, kwargs):
                # Pure non-device move (dtype / memory_format / non_blocking)
                # — пропускаем как есть.
                try:
                    return original_to(*args, **kwargs)
                except Exception:  # noqa: BLE001
                    return inner
            # Device-move detected: возвращаем original_to без device-args,
            # чтобы dtype/memory_format/non_blocking дошли до nn.Module.to.
            cleaned_args, cleaned_kwargs = _strip_device(args, kwargs)
            if not cleaned_args and not cleaned_kwargs:
                # Pure device move — наш split категорически воспрещает.
                return inner
            try:
                return original_to(*cleaned_args, **cleaned_kwargs)
            except Exception:  # noqa: BLE001
                return inner

        inner.to = _no_op_to  # type: ignore[method-assign]
        inner._ltx2_to_locked = True  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        # Если патч не навесился — sampler драгает DiT обратно,
        # graceful fallback (худший случай: OOM mid-sampling).
        pass


def _split_blocks_indices(strategy: str) -> tuple[int, ...]:
    """Возвращает индекс блока-разделителя для strategy.

    blocks_50_50 → 22 (блоки 0..21 → device0; 22..43 → device1)
    blocks_30_70 → 13 (блоки 0..12 → device0; 13..43 → device1)
    others        → None (whole-model moves)
    """
    if strategy == "blocks_50_50":
        return (22,)
    if strategy == "blocks_30_70":
        return (13,)
    return ()


def _build_patcher_for_load(unet_name: str, verbose: bool) -> Any:
    """Загружает GGUF через city96 UnetLoaderGGUF и возвращает оригинальный ModelPatcher."""
    if folder_paths is None:
        raise RuntimeError("folder_paths недоступен — пакет вне ComfyUI")

    full_path = folder_paths.get_full_path("diffusion_models", unet_name)
    if verbose:
        print(f"[ComfyUI-LTX2-MultiGPU] Loading {full_path}")

    cls = _get_city96_patcher_class()
    # City96 реализует load_unet() -> (MODEL_PATCHER,)
    instance = cls()
    load_fn = getattr(instance, "load_unet", None)
    if load_fn is None:
        # V3 API: define_schema / load_unet
        raise RuntimeError(
            f"{cls.__name__} не имеет load_unet() — несовместимая версия ComfyUI-GGUF"
        )

    primary_dev, _ = resolve_devices()
    (patcher,) = _call_with_ctx(primary_dev, load_fn, unet_name)

    if verbose:
        try:
            arch = patcher.model.__class__.__name__
        except Exception:  # noqa: BLE001
            arch = "?"
        print(f"[ComfyUI-LTX2-MultiGPU] Loaded arch={arch}")
    return patcher


def hybrid_split_gguf(
    gguf_name: str,
    strategy: str = "blocks_50_50",
    verbose: bool = False,
    donor_device: str = "auto",
) -> Any:
    """Главная точка: GGUF → ModelPatcher с раскиданными по GPU блоками.

    Реализация по PLAN §3.2:
      1. Загрузка через city96 UnetLoaderGGUF.load_unet (lazy GGMLOps).
      2. Классификация + .to(device) для каждого блока DiT.
      3. register_forward_hook на блоке-разделителе → hidden_states.to(dst_dev).
      4. Возврат ModelPatcher (R3-совместимый; LoRA/offload работают).

    Новые kwargs (UI mirror GemmaHybrid — см. commit message):
      donor_device ∈ {"auto","cuda:0","cuda:1","cpu"} — куда положить
                     "вторичную" половину DiT / целиком DiT.
                     CPU отврегается в INPUT_TYPES HybridSplitLoader
                     (DiT не иожет быть на CPU во время sampling'а); если
                     передан через programmatic call — WARN в verbose и движение
                     вторичной половины пропускается (рull DiT off GPU →
                     sampling stall). Используйте только cuda:auto/cuda:0/cuda:1.

      NB: `eject_models` НЕ поддерживается для DiT (anti-feature):
        - DiT вызывается каждый sampling-step. Offload DiT → CPU = PCIe
          bottleneck и sampling stall.
        - Risk #7 lock (.to() no-op) добавляет дополнительную причину: sampler
          не смог бы re-load DiT обратно на GPU после eject.

    Семантика donor_device по strategy:
      blocks_50_50 : primary_dev получает блоки [0..21] (22 шт),
                    donor_dev получает блоки [22..43] (22 шт).
    blocks_30_70 : primary_dev получает блоки [0..12] (13 шт, ~30%),
                    donor_dev получает блоки [13..43] (31 шт, ~70%).
                    forward_pre_hook на blocks[split_idx] двигает
                    hidden_states primary→donor.
      pipeline : весь DiT целиком на donor_dev (default auto→secondary).
      single_cuda0 : target=primary_dev (явный override, donor ignored).
      single_cuda1 : target=donor_dev (default auto→secondary, но user override побеждает).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}; allowed: {STRATEGIES}")

    if folder_paths is None or torch is None:
        raise RuntimeError(
            "hybrid_split_gguf требует ComfyUI runtime (folder_paths, torch)"
        )

    # ── Шаг 1: загрузка через city96 loader ─────────────────────────────────
    patcher = _build_patcher_for_load(gguf_name, verbose)

    # ── Шаг 2: target devices ───────────────────────────────────────────────
    primary_dev, secondary_dev = resolve_devices()
    donor_dev = resolve_donor_device(donor_device, primary_dev, secondary_dev)
    donor_is_cpu = str(donor_dev).startswith("cpu")

    # Defensive: cpu как donor для DiT — anti-feature. INPUT_TYPES HybridSplitLoader
    # уже фильтрует cpu, но programmatic call мог бы прокинуть — fallback на secondary.
    effective_donor = secondary_dev if donor_is_cpu else donor_dev
    if donor_is_cpu and verbose:
        print(
            "[ComfyUI-LTX2-MultiGPU] WARN: donor_device='cpu' отвергнут для DiT — "
            "DiT не может быть на CPU во время sampling'а. Fallback на secondary_dev."
        )

    is_pipeline = strategy == "pipeline"
    is_single0 = strategy == "single_cuda0"
    is_single1 = strategy == "single_cuda1"
    splits = _split_blocks_indices(strategy)

    # FIX MEDIUM_apply_strategy (final): degenerate guard в hybrid_split_gguf.
    # Нормализация применяется всегда + WARN печатается БЕЗ verbose_gate
    # (UX consistency с apply_strategy: один из двух loaders выводит visible
    # WARN, чтобы юзер не терялся в silent normalize).
    if (
        effective_donor == primary_dev
        and not is_single0
    ):
        print(
            "[ComfyUI-LTX2-MultiGPU] WARN: effective_donor==primary_dev "
            f"с strategy={strategy!r} → split дегенеративен "
            "(обе половины DiT коллапсируют на cuda:0). "
            "Нормализация → single_cuda0."
        )
        is_single0 = True
        is_single1 = False
        is_pipeline = False
        strategy = "single_cuda0"
        splits = ()

    # Strategy-логика (обновлена для donor_device):
    #   pipeline    → весь DiT на effective_donor (default auto→secondary=cuda:1)
    #   single_cuda0→ весь DiT на primary_dev (явное имя, donor ignored)
    #   single_cuda1→ весь DiT на effective_donor (user override над secondary)
    #   blocks_*    → split: [0..split_idx-1]@primary, [split_idx..N-1]@effective_donor
    if is_single0:
        target_dev = primary_dev
    elif is_single1 or is_pipeline:
        target_dev = effective_donor
    else:
        target_dev = None  # split mode — разные device для разных блоков

    if verbose:
        print(
            f"[ComfyUI-LTX2-MultiGPU] strategy={strategy} "
            f"primary={primary_dev} donor={effective_donor} (spec={donor_device!r})"
        )

    # ── Шаг 3: перенос блоков на нужные device ──────────────────────────────
    try:
        from comfy import model_management as mm  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("comfy.model_management недоступен") from exc

    inner = patcher.model  # nn.Module (UNetModel-like)
    diffusion = getattr(inner, "diffusion_model", inner)
    blocks = getattr(diffusion, "transformer_blocks", None)

    if target_dev is not None:
        # Whole-model move
        with mm.cuda_device_context(target_dev):
            inner.to(target_dev, non_blocking=False)
    elif blocks is None or len(blocks) != LTX2_DIT_BLOCK_COUNT:
        # fallback: оставляем на primary и предупреждаем
        if verbose:
            print(
                f"[ComfyUI-LTX2-MultiGPU] WARN: не нашёл 44 transformer_blocks "
                f"(нашёл {len(blocks) if blocks else 0}). "
                f"Возможно, неправильное GGUF-семейство — DiT остаётся на primary."
            )
        inner.to(primary_dev, non_blocking=False)
    else:
        # blocks split mode
        if not splits:
            inner.to(primary_dev, non_blocking=False)
        else:
            split_idx = splits[0]
            with mm.cuda_device_context(primary_dev):
                for i in range(0, split_idx):
                    blocks[i].to(primary_dev, non_blocking=False)
                # embed/head слои — на primary (flat named_modules, no recurse)
                _move_modules_with_prefix(
                    diffusion, primary_dev, *EMBED_AND_HEAD_REL
                )
                mm.soft_empty_cache()
            with mm.cuda_device_context(effective_donor):
                for i in range(split_idx, len(blocks)):
                    blocks[i].to(effective_donor, non_blocking=False)
                mm.soft_empty_cache()

            # srijithr forward_pre_hook на блоке split_idx —
            # двигает входной hidden_states с primary_dev на effective_donor
            # ДО forward на этом блоке. Не ломает autograd graph,
            # потому что hooks возвращает contract-modified inputs,
            # а не сам output.
            _remove_stored_hooks(patcher)
            handle = _install_cross_device_hook(
                blocks[split_idx], primary_dev, effective_donor
            )
            if handle is not None:
                _store_hook(patcher, handle)

    # ── Шаг 4: вернуть ModelPatcher с правильными meta ─────────────────────
    # Патчер city96 уже валидный ModelPatcher; мы лишь обновляем load_device и
    # offload_device, чтобы sampler знал, где искать веса. R3 device-management.
    try:
        with mm.cuda_device_context(primary_dev):
            patcher.load_device = primary_dev
            patcher.offload_device = primary_dev  # offload обратно на primary
    except Exception:  # noqa: BLE001
        # Если mm не даёт context — оставляем как есть
        patcher.load_device = primary_dev
        patcher.offload_device = primary_dev

    # Risk #7 fix: блокируем blanket ``inner.to(load_device)`` от ComfyUI
    # sampler'а, чтобы он не стянул split-блоки (cuda:1) обратно на primary
    # между итерациями sampling'а. Без этого вызов ``model_load()`` в начале
    # каждой KSampler-step утаскивает блоки на cuda:0 → OOM на 720p.
    _lock_inner_to(inner)

    # Meta-флаг для downstream-нод: «split применён»
    # Ассиметрия с ltx2_multigpu_gemma_split (Gemma) — DiT НЕ имеет поля
    # eject_models: DiT вызывается каждый sampling-step, offload DiT → CPU
    # = PCIe bottleneck и sampling stall. Forward_pre_hook остаётся primary→donor
    # даже если блоки коллапсировали на primary (N1 guard выше).
    try:
        # NEW (v0.2.1): "effective_donor" key добавлен рядом с legacy "secondary" /
        # "donor" для backward-compat. После MED-3 effective_donor может быть primary
        # или secondary в зависимости от donor_device widget — поэтому "secondary"
        # name misleading. Consumers должны читать "effective_donor" first.
        patcher.model_options["ltx2_multigpu_split"] = {
            "strategy": strategy,
            "primary": str(primary_dev),
            "effective_donor": str(effective_donor),  # canonical post-v0.2.1
            "secondary": str(secondary_dev),          # legacy (always = secondary)
            "donor": str(effective_donor),            # legacy alias effective_donor
            "donor_spec": donor_device,
            "block_split_index": splits[0] if splits else None,
        }
    except Exception:  # noqa: BLE001
        pass

    if verbose:
        # Per-component allocation log для DiT HybridSplit (file size + target echo).
        try:
            import os
            full_path = folder_paths.get_full_path("diffusion_models", gguf_name) if folder_paths else None
            unet_gb = os.path.getsize(full_path) / (1024 ** 3) if full_path else 0.0
        except Exception:  # noqa: BLE001
            unet_gb = 0.0
        try:
            print(
                f"[ComfyUI-LTX2-MultiGPU] dit_alloc = {unet_gb:.2f} GB file @ "
                f"target={target_dev or effective_donor} "
                f"(strategy={strategy} donor={donor_device!r})"
            )
        except Exception:  # noqa: BLE001
            pass
        # VRAM free after load (de-dup по индексам primary/donor)
        try:
            if torch.cuda.is_available():
                seen_idx: set[int] = set()
                for d in [primary_dev, effective_donor]:
                    if d.type != "cuda":
                        continue
                    idx = int(d.index)
                    if idx in seen_idx:
                        continue
                    seen_idx.add(idx)
                    free, total = torch.cuda.mem_get_info(idx)
                    print(
                        f"[ComfyUI-LTX2-MultiGPU] cuda:{idx} "
                        f"free={free / 1024 ** 3:.2f} GB "
                        f"total={total / 1024 ** 3:.2f} GB"
                    )
        except Exception:  # noqa: BLE001
            pass

    return patcher


def _store_hook(patcher: Any, handle: Any) -> None:
    """Сохраняет forward_pre_hook handle на patcher (per-patcher, не global).

    Если patcher не позволяет setattr — молча отбрасываем (hook останется
    активным до GC модели, но race condition в multi-prompt уменьшается).
    """
    try:
        try:
            stored = patcher._ltx2_hooks
        except AttributeError:
            stored = []
            patcher._ltx2_hooks = stored  # type: ignore[attr-defined]
        stored.append(handle)
    except Exception:  # noqa: BLE001
        # Не fatale — hook просто не будет очищен через этот path
        pass


def _remove_stored_hooks(patcher: Any) -> None:
    """Отменяет все ранее установленные hooks перед новым split."""
    try:
        stored = list(getattr(patcher, "_ltx2_hooks", []) or [])
    except Exception:  # noqa: BLE001
        stored = []
    for handle in stored:
        try:
            handle.remove()
        except Exception:  # noqa: BLE001
            pass
    try:
        patcher._ltx2_hooks = []  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


# NOTE: глобальная _LTX2_HOOK_REGISTRY РАнее создавалась как fallback,
# но привела к race condition при нескольких patcher в одной сессии —
# УДАЛЕНА. Используем ТОЛЬКО per-patcher setattr через _store_hook.


# Relative qualnames (от корня ``diffusion`` модуля), матчащиеся
# flat-циклом ``named_modules()``. ``transformer_blocks.*`` намеренно
# отсутствует — блоки DiT обрабатываются отдельным циклом в
# ``hybrid_split_gguf`` / ``apply_strategy``.
EMBED_AND_HEAD_REL: tuple[str, ...] = (
    "time_embed",
    "adaln",
    "adaln_single",
    "patchify_proj",
    "proj_in",
    "norm_in",
    "proj_out",
    "norm_out",
)


def _move_modules_with_prefix(
    parent: Any, device: Any, *rel_prefixes: str
) -> int:
    """Двигает на ``device`` children ``parent``, чей qualname равен
    ровно одному из ``rel_prefixes`` ИЛИ начинается с
    ``rel_prefix + "."``.

    Один проход ``parent.named_modules()`` — никакого рекурсивного dive
    и никакого match'а по под-сегментам. На 22B DiT это на порядок
    быстрее и безопаснее прежнего ``_move_attrs_by_prefix``.
    Возвращает количество уникально перенесённых submodules.

    NB: ``transformer_blocks.*`` намеренно НЕ матчится — блоки DiT
    идут через отдельный цикл в ``hybrid_split_gguf``.
    """
    if torch is None:
        return 0
    moved_ids: set[int] = set()
    count = 0
    for name, module in parent.named_modules():
        if not name:  # пропустить сам parent
            continue
        matched = False
        for rpfx in rel_prefixes:
            if name == rpfx or name.startswith(rpfx + "."):
                matched = True
                break
        if not matched:
            continue
        mid = id(module)
        if mid in moved_ids:
            continue  # идемпотентность: один nn.Module может встретиться
            # несколько раз в named_modules() (наследуемые подмодули)
        try:
            module.to(device, non_blocking=False)
            moved_ids.add(mid)
            count += 1
        except Exception:  # noqa: BLE001
            pass
    return count


def _looks_like_proj_path(name: str) -> bool:
    """Распознаёт alternative naming для text_projection в Gemma pipeline.

    Examples:
      - "model.text_projection.weight"
      - "model.proj.weight"  (some Gemma variants)
      - "text_projection.weight"
      - "model.projection.weight"
      - "proj_out.weight"
    """
    n = name.lower()
    return (
        n.startswith("text_projection.")
        or n.startswith("proj_out.")
        or n.startswith("model.projection.")
        or n.startswith("model.proj.")
        or "/projection/" in n
        or ".projection." in n
        or n.endswith(".proj")
    )


def _move_param(p: Any, device: Any) -> bool:
    """FIX LOW #5 + MEDIUM #3: in-place move ``nn.Parameter`` в ``device``.

    Использует ``torch.no_grad()`` чтобы autograd не path'ил data reassign.
    Возвращает True ТОЛЬКО если torch доступен И move успешен.
    Defensive при ``torch is None`` — мы за пределами ComfyUI runtime,
    и inference невозможен; статический return False здесь OK.
    """
    if torch is None:
        return False
    try:
        with torch.no_grad():
            p.data = p.data.to(device, non_blocking=False)
        return True
    except Exception:  # noqa: BLE001
        return False


def _move_buffer(b: Any, device: Any) -> bool:
    """FIX LOW #5 + MEDIUM #3: in-place move registered buffer в ``device``."""
    if torch is None:
        return False
    try:
        with torch.no_grad():
            b.data = b.data.to(device, non_blocking=False)
        return True
    except Exception:  # noqa: BLE001
        return False


def load_gemma_hybrid(
    encoder_name: str,
    projection_name: str,
    verbose: bool = False,
    donor_device: str = "auto",
    eject_models: bool = False,
) -> Any:
    """Gemma 12B FP4 → donor_device, text_projection → cuda:0, wrapped в ModelPatcher.

    PLAN §3.4 / MODEL_FACTS §6:
        - text encoder (≈7.5 GB FP4) → donor_device
        - text_projection (≈2.15 GB) → primary_dev (cuda:0)

    Новые kwargs (UI mirror DualCLIPLoaderDisTorch2MultiGPU):
      donor_device ∈ {"auto","cuda:0","cuda:1","cpu"} — куда грузить encoder.
                     auto ⇒ secondary (default cuda:1 в dual-GPU config).
                     cpu  ⇒ encoder остаётся на CPU; параметры text_projection
                            всё равно поднимаются на primary для sampling'а.
      eject_models=True ⇒ patcher.offload_device=CPU + mm.soft_empty_cache().
                     NB: под Risk #7 lock (.to() no-op) sampler не сможет
                         blanket-поднять параметры projection обратно на cuda:0
                         после eject ⇒ возможен лёгкий re-conditioning stall.
                         Используйте только если уверены.

    Использует встроенный `comfy.sd.load_clip(ckpt_paths=[enc, proj])` для
    Gemma3/Gemma4 (master ComfyUI c поддержкой с late 2025). При успешной
    загрузке делит веса через named_parameters/named_buffers walk и добавляет
    forward_pre_hook на parent-module text_projection для переноса
    hidden_states donor_dev→primary_dev (srijithr-паттерн).

    Returns: ModelPatcher-совместимый объект (R3, маркирован для sampler).

    Raises:
        RuntimeError: если ComfyUI API не поддерживает Gemma / FP4; если
            safetensors не распознаны; если Gemma encoder не найден в
            `text_encoders/`.
    """
    if folder_paths is None or torch is None:
        raise RuntimeError("load_gemma_hybrid требует ComfyUI runtime")

    primary_dev, secondary_dev = resolve_devices()
    donor_dev = resolve_donor_device(donor_device, primary_dev, secondary_dev)
    donor_is_cpu = (str(donor_dev).startswith("cpu"))

    if verbose:
        print(
            f"[ComfyUI-LTX2-MultiGPU] GemmaHybrid: encoder={encoder_name} @ "
            f"{donor_dev} (donor_device={donor_device!r}), "
            f"projection={projection_name} @ {primary_dev}, "
            f"eject_models={eject_models}"
        )

    # Eject + Risk #7 lock conflict warning.
    if eject_models and verbose:
        print(
            "[ComfyUI-LTX2-MultiGPU] WARN: eject_models=True активирует "
            "offload_device=CPU. NB: под Risk #7 lock (.to() no-op) sampler "
            "не сможет blanket re-load projection на cuda:0 после eject. "
            "Если сценарий — single-pass encoding без re-conditioning, "
            "OK; иначе оставьте False."
        )

    # ── Шаг 1: импорт comfy internals (R6: тяжёлые импорты в try/except) ─────
    try:
        from comfy import sd as comfy_sd  # type: ignore[import-not-found]
        from comfy import model_management as mm  # type: ignore[import-not-found]
        from comfy import model_patcher as comfy_mp  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"comfy sd/model_management/model_patcher недоступны: {exc}"
        ) from exc

    # ── Шаг 2: получить file paths из ComfyUI folder_paths ───────────────────
    # BUG-2 fix: encoder берём из text_encoders/ (стандарт для Gemma 3/4
    # safetensors). Projection ИЩЕМ в text_encoders/ ПЕРВЫМ, fallback в clip/
    # (для случаев когда projection лежит в clip/ — стандартный путь для
    # city96/ComfyUI-GGUF). Раньше был хардкод только text_encoders — если
    # юзер по проекту city96 клал projection в clip/, load_gemma_hybrid
    # падал с FileNotFoundError несмотря на корректный dropdown вход.
    enc_path = folder_paths.get_full_path("text_encoders", encoder_name)
    proj_path = (
        folder_paths.get_full_path("text_encoders", projection_name)
        or folder_paths.get_full_path("clip", projection_name)
    )
    if not enc_path:
        raise FileNotFoundError(
            f"Gemma encoder '{encoder_name}' не найден в text_encoders/"
        )
    if not proj_path:
        raise FileNotFoundError(
            f"text_projection '{projection_name}' не найден ни в text_encoders/ "
            f"ни в clip/"
        )

    # ── Шаг 3: build CLIP через comfy.sd.load_clip(ckpt_paths=[enc, proj]) ──
    # Если load_clip отсутствует — клиент на старой версии ComfyUI; raise с
    # понятным сообщением.
    if not hasattr(comfy_sd, "load_clip"):
        raise RuntimeError(
            "comfy.sd.load_clip отсутствует — требуется современный ComfyUI "
            "с Gemma3/Gemma4 support. Обновитесь: cd ComfyUI && git pull"
        )

    try:
        clip_obj = comfy_sd.load_clip(
            ckpt_paths=[enc_path, proj_path],
            embedding_directory=None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"comfy.sd.load_clip failed for Gemma hybrid split: {exc}. "
            f"Возможные причины: "
            f"(a) FP4 weights без comfy.ops FP4 support; "
            f"(b) safetensors не из Gemma 3 / Gemma 4 family; "
            f"(c) ComfyUI version ниже Gemma3 merge (late-2025 master)."
        ) from exc
    if clip_obj is None:
        raise RuntimeError(
            "comfy.sd.load_clip вернул None — Gemma не распознана ни по одному из "
            "файлов. Проверьте, что encoder_name — Gemma3/4 safetensors, "
            "projection_name — отдельный файл или integrated layer."
        )

    # ── Шаг 4: parameter-level split (text_projection@primary, rest@secondary) ──
    # Parameter-level move() надёжнее module-level: nested nn.Module hierarchy
    # не мешает разделению. Каждый параметр move()'ed отдельно под правильным
    # cuda_device_context — PCIe-safe без race-conditions.
    patcher_obj = clip_obj.patcher if hasattr(clip_obj, "patcher") else clip_obj
    inner = (
        patcher_obj.model  # type: ignore[union-attr]
        if hasattr(patcher_obj, "model") else patcher_obj
    )
    if inner is None:
        raise RuntimeError(
            "comfy.sd.load_clip вернул CLIP без .model attribute — невозможно "
            "применить device split. ComfyUI master API contract нарушен."
        )

    proj_param_count = 0
    encoder_param_count = 0
    proj_move_failures: list[str] = []

    # Parametры projection → primary_dev.
    # FIX HIGH (silent projection move failure): если proj-параметр НЕ
    # переехал на primary_dev (torch=None или device-mismatch), pre-hook будет
    # двигать hidden_states на cuda:0, а proj-weights останутся на CPU/donor →
    # неприятный RuntimeError в середине sampling-loop. Собираем fail-list и
    # ПОСЛЕ цикла raise'им с понятным именем параметра и возможной причиной.
    # Skip raise для тестовых сред без torch (HF #1: verbose-gated WARN не
    # защищает prod — upstream kingjin не знал что мы стоим на cpu device).
    with mm.cuda_device_context(primary_dev):
        for pname, p in list(inner.named_parameters()):
            if "text_projection" not in pname.lower() and not _looks_like_proj_path(pname):
                continue
            if _move_param(p, primary_dev):
                proj_param_count += 1
            else:
                proj_move_failures.append(pname)

    # Raise если есть failures и torch доступен.
    # (При torch=None полная silent-деградация — мы вне ComfyUI, явный raise
    # зашумит тестсьют; User-runner увидит чистую ошибку через ``verbose_log``.)
    if proj_move_failures and torch is not None:
        sample_names = ", ".join(repr(n) for n in proj_move_failures[:5])
        more = (
            f" и ещё {len(proj_move_failures) - 5}"
            if len(proj_move_failures) > 5 else ""
        )
        raise RuntimeError(
            f"load_gemma_hybrid: не удалось перенести {len(proj_move_failures)} "
            f"text_projection параметров на {primary_dev} (sample: {sample_names}{more}). "
            f"Проверьте свободную VRAM на primary и что donor_device != 'cpu' "
            f"для параметров проекции. Sampling с half-moved projection "
            f"крашится с confusing 'device mismatch' на cuda:0."
        )
    elif proj_move_failures and verbose:
        print(
            f"[ComfyUI-LTX2-MultiGPU] WARN: {len(proj_move_failures)} proj params "
            f"не перенесены (torch=None, тест-окружение): "
            f"{[n for n in proj_move_failures[:3]]}…"
        )

    # Parametры encoder → donor_dev
    # Если donor_device='cpu' — пропускаем move: encoder остаётся на исходном
    # device, куда его положил comfy.sd.load_clip (CPU by default).
    if not donor_is_cpu:
        with mm.cuda_device_context(donor_dev):
            for pname, p in list(inner.named_parameters()):
                if "text_projection" in pname.lower() or _looks_like_proj_path(pname):
                    continue
                if _move_param(p, donor_dev):
                    encoder_param_count += 1
                elif verbose:
                    print(
                        f"[ComfyUI-LTX2-MultiGPU] WARN: encoder param {pname!r} → "
                        f"{donor_dev} failed (torch=None или device mismatch)"
                    )
    # NOTE (v0.2.2): старый ``elif verbose: encoder_param_count = sum(...); print(...)``
    # УДАЛЁН — его функционал поглощён unconditional-блоком ниже (FIX LOW #3).

    # Buffers (key embedding_table, registered buffers и т.д.)
    with mm.cuda_device_context(primary_dev):
        for bname, b in list(inner.named_buffers()):
            if "text_projection" not in bname.lower() and not _looks_like_proj_path(bname):
                continue
            _move_buffer(b, primary_dev)
    if not donor_is_cpu:
        with mm.cuda_device_context(donor_dev):
            for bname, b in list(inner.named_buffers()):
                if "text_projection" in bname.lower() or _looks_like_proj_path(bname):
                    continue
                _move_buffer(b, donor_dev)

    # FIX LOW #3 (encoder_param_count unconditional): даже если verbose=False
    # и donor=cpu, downstream-ноды (сэмплер debug-info, MID-7) могут читать
    # `final_patcher.model_options["ltx2_multigpu_gemma_split"]["encoder_param_count"]`.
    # Раньше эта переменная считалась ТОЛЬКО внутри `elif verbose:` — на
    # silent-path она оставалась = 0 → ложный отчёт '0 энкодерных параметров'.
    # Считаем всегда (дешёвый named_parameters walk; порядка ms).
    if donor_is_cpu:
        # FIX PRODUCTION tuple-unpack bug: named_parameters() yields tuples
        # ``(name, tensor)``; ``_.lower()`` падало с AttributeError. Распаковываем.
        encoder_param_count = sum(
            1 for name, _p in inner.named_parameters()
            if "text_projection" not in name.lower()
            and not _looks_like_proj_path(name)
        )
        if verbose:
            print(
                f"[ComfyUI-LTX2-MultiGPU] donor_device=cpu — {encoder_param_count} "
                f"encoder params остаются на исходном device (без .to())."
            )

    if verbose:
        print(
            f"[ComfyUI-LTX2-MultiGPU] Split: encoder={encoder_param_count} params "
            f"@ {donor_dev}, proj={proj_param_count} params @ {primary_dev}"
        )

    # ── Шаг 5: forward_pre_hook на proj parent (cuda:1→cuda:0 hidden_states) ─
    # GemmaPipeline forward call sequence:
    #   embed_tokens (cuda:1) → layers 0..N (cuda:1) → text_projection (cuda:0)
    # Последний hidden_states выходит с cuda:1 (encoder), proj ждёт с cuda:0.
    # Без hook'а PyTorch кинет RuntimeError при .to() mismatch.
    def _move_inputs_to(obj: Any) -> Any:
        """FIX CRITICAL #1 + MEDIUM #3: рекурсивный move по Tensor/tuple/list/dict/set.

        kwargs для forward_pre_hook(PyTorch 2.0+ with_kwargs=True) тоже поддерживаются
        (через вызывающий _proj_pre_hook).
        """
        if torch is None:
            return obj
        if isinstance(obj, torch.Tensor):
            if obj.device == primary_dev:
                return obj
            try:
                return obj.to(primary_dev, non_blocking=True)
            except Exception:  # noqa: BLE001
                return obj.to(primary_dev)
        if isinstance(obj, (tuple, list)):
            return type(obj)(_move_inputs_to(x) for x in obj)
        if isinstance(obj, dict):
            return {k: _move_inputs_to(v) for k, v in obj.items()}
        if isinstance(obj, set):
            return {_move_inputs_to(x) for x in obj}
        return obj

    def _proj_pre_hook(_mod: Any, args: Any, kwargs: Any = None) -> Any:
        """FIX CRITICAL #1 + MEDIUM #2: всегда возвращаем tuple ``(args, kwargs)``."""
        moved_args = _move_inputs_to(args)
        moved_kwargs = _move_inputs_to(kwargs) if kwargs is not None else {}
        return moved_args, moved_kwargs

    # FIX MEDIUM #3: ищем САМ модуль `text_projection` (не его parent).
    # parent-search хрупок — если ComfyUI завернёт proj в дополнительный layer,
    # hook не навесится. Прямой подвес на сам модуль проще и устойчивее.
    proj_module = None
    if torch is not None:
        for modname, mod in inner.named_modules():
            if modname == "text_projection" or modname.endswith(".text_projection"):
                proj_module = mod
                break

    # FIX CRITICAL (hook accumulation + leak path, v0.2.2 polish): ComfyUI
    # кеширует результат ``comfy.sd.load_clip`` и при повторных вызовах
    # load_gemma_hybrid через эту ноду мы получаем тот же ``inner``. Без
    # очистки каждый вызов навешивает НОВЫЙ forward_pre_hook на
    # ``proj_module._forward_pre_hooks`` → O(n) хуков на n-м вызове.
    #
    # ВАЖНО leak-path fix (reviewer v0.2.2): cleanup ОБЯЗАН работать даже если
    # proj_module is None (Gemma wrapper без прямого text_projection модуля).
    # Per-patcher handle list и nn.Module-level _forward_pre_hooks дикты
    # разные — оба чистятся здесь безусловно; proj_module-specific чистится
    # только когда proj_module есть.
    _remove_stored_hooks(patcher_obj)
    try:
        hd = getattr(inner, "_forward_pre_hooks", None)
        if hd is not None:
            # Dict[OrderedDict[int, RemovableHandle]] — собрать ключи заранее,
            # иначе удаление в итерации может крэшнуть.
            for k in list(hd.keys()):
                try:
                    hd[k].remove()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    del hd[k]
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass
    if proj_module is not None:
        try:
            hd = getattr(proj_module, "_forward_pre_hooks", None)
            if hd is not None:
                for k in list(hd.keys()):
                    try:
                        hd[k].remove()
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        del hd[k]
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass

    if (
        proj_module is not None
        and torch is not None
        and hasattr(proj_module, "register_forward_pre_hook")
    ):
        try:
            handle = proj_module.register_forward_pre_hook(
                _proj_pre_hook, with_kwargs=True
            )
        except TypeError:
            # PyTorch < 2.0 fallback.
            handle = proj_module.register_forward_pre_hook(
                lambda _m, a: _move_inputs_to(a)
            )
        if handle is not None:
            try:
                _store_hook(patcher_obj, handle)
            except Exception:  # noqa: BLE001
                pass  # hook останется активным до GC — acceptable fallback

    # ── Шаг 6: final ModelPatcher wrap (R3 R4) ──────────────────────────────
    offload_dev = torch.device("cpu") if eject_models else primary_dev
    if hasattr(patcher_obj, "load_device") and hasattr(patcher_obj, "model_options"):
        final_patcher = patcher_obj
        try:
            final_patcher.load_device = primary_dev
            final_patcher.offload_device = offload_dev
        except Exception:  # noqa: BLE001
            pass
    else:
        # Fallback: завернуть в новый ModelPatcher (defensive, не должен
        # срабатывать в стандартном ComfyUI пути)
        final_patcher = comfy_mp.ModelPatcher(
            model=inner,
            load_device=primary_dev,
            offload_device=offload_dev,
        )

    # Meta для downstream-нод (для sampler-ноды "see split info")
    try:
        final_patcher.model_options["ltx2_multigpu_gemma_split"] = {
            "encoder": str(donor_dev),
            "encoder_spec": donor_device,
            "projection": str(primary_dev),
            "encoder_param_count": encoder_param_count,
            "proj_param_count": proj_param_count,
            "eject_models": eject_models,
            "offload_device": str(offload_dev),
            "model_class": type(inner).__name__,
        }
    except Exception:  # noqa: BLE001
        pass

    # Risk #7 fix: блокируем .to() чтобы ``comfy.sd.load_clip`` или sampler
    # не «упростили» hybrid projection+encoder split при следующей загрузке
    # (Gemma encoder всё ещё на cuda:1, projection на cuda:0).
    _lock_inner_to(inner)

    mm.soft_empty_cache()

    if verbose:
        # Per-component allocation log
        # Используем file size напрямую (dual-fp4/bf16 storages варьируются,
        # но наш contract: encoder@donor, proj@primary, offload=eject?cpu:primary).
        try:
            import os
            enc_gb = os.path.getsize(enc_path) / (1024 ** 3) if enc_path else 0.0
            proj_gb = os.path.getsize(proj_path) / (1024 ** 3) if proj_path else 0.0
        except Exception:  # noqa: BLE001
            enc_gb = proj_gb = 0.0
        print(
            f"[ComfyUI-LTX2-MultiGPU] encoder_alloc = {enc_gb:.2f} GB file @ "
            f"{donor_dev} ({encoder_param_count} params moved)"
        )
        print(
            f"[ComfyUI-LTX2-MultiGPU] proj_alloc = {proj_gb:.2f} GB file @ "
            f"{primary_dev} ({proj_param_count} params moved)"
        )
        print(
            f"[ComfyUI-LTX2-MultiGPU] eject_models={eject_models} "
            f"offload_target={offload_dev}"
        )
        # VRAM free after load
        try:
            if torch.cuda.is_available():
                seen_idx: set[int] = set()
                for d in [primary_dev, donor_dev]:
                    if d.type != "cuda":
                        continue
                    idx = int(d.index)
                    if idx in seen_idx:
                        continue
                    seen_idx.add(idx)
                    free, total = torch.cuda.mem_get_info(idx)
                    print(
                        f"[ComfyUI-LTX2-MultiGPU] cuda:{idx} "
                        f"free={free / 1024 ** 3:.2f} GB "
                        f"total={total / 1024 ** 3:.2f} GB"
                    )
        except Exception:  # noqa: BLE001
            pass

    return final_patcher


def apply_strategy(
    patcher: Any,
    strategy: str,
    verbose: bool = False,  # noqa: ARG003
    donor_device: str = "auto",
) -> Any:
    """Применяет новую стратегию к уже загруженному ModelPatcher.

    ⚠️ **DiT-only**: предполагается, что ``patcher`` — это DiT ModelPatcher от
    ``hybrid_split_gguf`` (44 transformer_blocks + diffusion_model attribute +
    time_embed/adaln/etc outer layers). Для Gemma encoder patcher'а от
    ``load_gemma_hybrid`` эта функция **НЕ предназначена** — Gemma split
    (text_projection@primary + encoder@donor) делается в
    ``load_gemma_hybrid`` и не подлежит hot-swap через apply_strategy. Если
    в будущем понадобится apply_strategy для Gemma — нужна отдельная функция
    с проекцией на Gemma layout (без ``_move_modules_with_prefix`` от
    diffusion_model, без ``transformer_blocks`` lookup).

    NB: внутри использует cache модель — повторное перемещение блоков между
    GPU. Удаляет старые forward_hook'и через .modules() и пересоздаёт.

    Args:
        patcher: уже-загруженный DiT ModelPatcher (от ``hybrid_split_gguf``).
        strategy: одна из ``STRATEGIES``.
        verbose: deprecated — degenerate WARN печатается безусловно для UX
                 consistency с hybrid_split_gguf. Param сохранён в сигнатуре
                 для backward-compat c LTX2_MultiGPU_DeviceStrategy нодой.
        donor_device: куда класть вторичную половину DiT / целиком DiT в
                      pipeline / single_cuda1. Семантика зеркалирует
                      ``hybrid_split_gguf``:
                        - "auto"  → secondary_dev (default cuda:1)
                        - "cuda:0"/"cuda:1" → raw override
                        - "cpu"   → fallback на secondary_dev с WARN
                                      (DiT не может жить на CPU между
                                      sampling-шагами). Новый c v0.2.1 —
                                      раньше hardcoded был secondary_dev.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}; allowed: {STRATEGIES}")

    if patcher is None:
        raise RuntimeError("apply_strategy: patcher == None")

    try:
        from comfy import model_management as mm
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("comfy.model_management недоступен") from exc

    inner = patcher.model
    diffusion = getattr(inner, "diffusion_model", inner)
    blocks = getattr(diffusion, "transformer_blocks", None)

    # Очистить старые hook'и (srijithr pattern: pre_hook на блоке split_idx).
    # Важно: НЕ мутировать dict во время итерации — собираем в list.
    for m in [inner, diffusion, *(blocks or [])]:
        for hook_dict_name in ("_forward_pre_hooks", "_forward_hooks"):
            try:
                hdict = getattr(m, hook_dict_name, None)
            except Exception:  # noqa: BLE001
                hdict = None
            if hdict is None:
                continue
            try:
                # OK для Dict[OrderedDict] — собрать ключи в list, потом удалять
                for k in list(hdict.keys()):
                    handle = hdict[k]
                    try:
                        handle.remove()
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        del hdict[k]
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass
    _remove_stored_hooks(patcher)

    primary_dev, secondary_dev = resolve_devices()
    # NEW (v0.2.1): donor_device resolve — раньше был хардкод secondary_dev.
    donor_dev = resolve_donor_device(donor_device, primary_dev, secondary_dev)
    donor_is_cpu = str(donor_dev).startswith("cpu")
    effective_donor = secondary_dev if donor_is_cpu else donor_dev
    if donor_is_cpu:
        # NEW (v0.2.1): WARN о cpu как donor для DiT — UNCONDITIONAL для
        # UX consistency с hybrid_split_gguf. legacy-strategy-switch без
        # verbose=LIVE давал silent fallback раньше.
        print(
            "[ComfyUI-LTX2-MultiGPU] WARN: apply_strategy: donor_device='cpu' "
            "отвергнут для DiT — fallback на secondary_dev."
        )
    splits = _split_blocks_indices(strategy)
    is_pipeline = strategy == "pipeline"
    is_single0 = strategy == "single_cuda0"
    is_single1 = strategy == "single_cuda1"

    # FIX MEDIUM_apply_strategy (final): degenerate guard в apply_strategy.
    # Нормализация применяется всегда + WARN печатается БЕЗ verbose_gate.
    # Silent normalization = bad UX (юззер не знает что его split дегенерировал
    # в single_cuda0 на single-GPU setup). Консистентно с hybrid_split_gguf.
    if (
        primary_dev == effective_donor
        and not is_single0
    ):
        print(
            "[ComfyUI-LTX2-MultiGPU] WARN: apply_strategy: effective_donor==primary_dev "
            f"с strategy={strategy!r} → split дегенеративен "
            "(обе половины DiT коллапсируют на cuda:0). "
            "Нормализация → single_cuda0."
        )
        is_single0 = True
        is_single1 = False
        is_pipeline = False
        strategy = "single_cuda0"
        splits = ()

    if is_single0:
        target_dev = primary_dev
    elif is_single1 or is_pipeline:
        target_dev = effective_donor
    else:
        target_dev = None

    # FIX apply_strategy (b)+(e)+(d): bypass _lock_inner_to патч через
    # _ltx2_original_to (Risk #7 lock блокирует device-перемещения от sampler'а,
    # но apply_strategy — это legitimate strategy-switch, не sampler вызов).
    # Также carry EMBED_AND_HEAD_REL @ primary при split switch (могли остаться
    # на cuda:1 после whole-model стратегии) + defensive else-fallback для
    # пустых splits (избегаем silent no-op при добавлении новых strategy).
    _inner_original_to = getattr(inner, "_ltx2_original_to", inner.to)

    if target_dev is not None:
        with mm.cuda_device_context(target_dev):
            _inner_original_to(target_dev, non_blocking=False)
    elif blocks is not None and splits:
        split_idx = splits[0]
        with mm.cuda_device_context(primary_dev):
            for i in range(0, split_idx):
                blocks[i].to(primary_dev)
            # FIX (b): carry embed/head layers @ primary при split switch.
            # Идемпотентно — если уже @ primary, _move_modules_with_prefix
            # делает no-op move (cost: один named_modules() walk).
            _move_modules_with_prefix(
                diffusion, primary_dev, *EMBED_AND_HEAD_REL
            )
        with mm.cuda_device_context(effective_donor):
            for i in range(split_idx, len(blocks)):
                blocks[i].to(effective_donor)
        _remove_stored_hooks(patcher)
        handle = _install_cross_device_hook(
            blocks[split_idx], primary_dev, effective_donor
        )
        if handle is not None:
            _store_hook(patcher, handle)
    else:
        # FIX (d): defensive fallback — split strategy с пустым splits
        # (новые strategy в STRATEGIES без _split_blocks_indices branch;
        # degenerate single-GPU + non-single0 после normalize) → whole-model
        # move @ primary, чтобы избежать silent no-op → runtime cross-device
        # OOM / device mismatch.
        # UNCONDITIONAL WARN (per UX consistency с hybrid_split_gguf
        # degenerate-guard и single-GPU normalize в apply_strategy выше —
        # оба verbose-unconditional). verbose_gate здесь бы скрыл silent
        # no-op от пользователей, что defeats purpose of defensive branch.
        print(
            "[ComfyUI-LTX2-MultiGPU] WARN: apply_strategy: "
            f"strategy={strategy!r} вернул splits={splits} и "
            "target_dev=None — fallback на whole-model move @ "
            f"primary_dev={primary_dev}"
        )
        with mm.cuda_device_context(primary_dev):
            _inner_original_to(primary_dev, non_blocking=False)

    try:
        # NEW (v0.2.1): effective_donor как canonical key, "secondary" оставлен как
        # legacy alias для backward-compat. После MED-3 effective_donor может быть
        # как primary, так и secondary — поэтому "secondary" name misleading.
        patcher.model_options["ltx2_multigpu_split"] = {
            "strategy": strategy,
            "primary": str(primary_dev),
            "effective_donor": str(effective_donor),  # canonical post-v0.2.1
            "secondary": str(secondary_dev),          # legacy (always = secondary)
            "donor_spec": donor_device,
            "block_split_index": splits[0] if splits else None,
        }
    except Exception:  # noqa: BLE001
        pass

    # Risk #7 fix: при смене стратегии lock .to() надо переустановить.
    # Если до этого был разный inner ref (apply_strategy с patcher от другой
    # node-call) — без явного вызова sampler стянет блоки обратно.
    _lock_inner_to(inner)

    mm.soft_empty_cache()
    return patcher
