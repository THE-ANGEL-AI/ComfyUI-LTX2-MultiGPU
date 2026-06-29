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


def _get_city96_patcher_class():
    """Импортирует UnetLoaderGGUF (city96) лениво, чтобы R6 не падать.

    Совместим с загрузкой обоих имён: UnetLoaderGGUF (новая) / unet_loader_gguf (старая).
    """
    candidates = [
        # Популярные пути в custom_nodes/ComfyUI-GGUF
        ("nodes", "UnetLoaderGGUF"),
        ("unet_loader_gguf", "UnetLoaderGGUF"),
        ("nodes", "UNETLoaderGGUF"),
    ]
    last_exc: Exception | None = None
    for mod_name, cls_name in candidates:
        try:
            module = __import__(mod_name)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
        cls = getattr(module, cls_name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        "city96 ComfyUI-GGUF не найден; клонируйте "
        "https://github.com/city96/ComfyUI-GGUF в custom_nodes/"
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
        return obj

    def _pre_hook(_mod: Any, inputs: Any) -> Any:
        return _move(inputs)

    handle = module.register_forward_pre_hook(_pre_hook)
    return handle


def _split_blocks_indices(strategy: str) -> tuple[int, ...]:
    """Возвращает индекс блока-разделителя для strategy.

    blocks_50_50 → 22 (блоки 0..21 → device0; 22..43 → device1)
    blocks_30_70 → 14 (блоки 0..13 → device0; 14..43 → device1)
    others        → None (whole-model moves)
    """
    if strategy == "blocks_50_50":
        return (22,)
    if strategy == "blocks_30_70":
        return (14,)
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
    gguf_name: str, strategy: str = "blocks_50_50", verbose: bool = False
) -> Any:
    """Главная точка: GGUF → ModelPatcher с раскиданными по GPU блоками.

    Реализация по PLAN §3.2:
      1. Загрузка через city96 UnetLoaderGGUF.load_unet (lazy GGMLOps).
      2. Классификация + .to(device) для каждого блока DiT.
      3. register_forward_hook на блоке-разделителе → hidden_states.to(dst_dev).
      4. Возврат ModelPatcher (R3-совместимый; LoRA/offload работают).
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
    is_pipeline = strategy == "pipeline"
    is_single0 = strategy == "single_cuda0"
    is_single1 = strategy == "single_cuda1"
    splits = _split_blocks_indices(strategy)

    # В ТЕКУЩЕЙ версии мы работаем через обёртку ModelPatcher.
    # Strategy-логика:
    #   pipeline    → весь DiT на secondary_dev (cuda:1)
    #   single_cuda0→ весь DiT на primary_dev (cuda:0)
    #   single_cuda1→ весь DiT на secondary_dev (cuda:1)
    #   blocks_*    → split по выбранной границе
    if is_single0:
        target_dev = primary_dev
    elif is_single1 or is_pipeline:
        target_dev = secondary_dev
    else:
        target_dev = None  # split mode — разные device для разных блоков
    if verbose:
        print(
            f"[ComfyUI-LTX2-MultiGPU] strategy={strategy} "
            f"primary={primary_dev} secondary={secondary_dev}"
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
                # embed/head слои — на primary
                for name_prefix in DIFFUSION_EMBED_PREFIXES:
                    _move_attrs_by_prefix(diffusion, name_prefix, primary_dev)
                _move_attrs_by_prefix(
                    diffusion, "model.diffusion_model.proj_out.", primary_dev
                )
                _move_attrs_by_prefix(
                    diffusion, "model.diffusion_model.norm_out.", primary_dev
                )
                mm.soft_empty_cache()
            with mm.cuda_device_context(secondary_dev):
                for i in range(split_idx, len(blocks)):
                    blocks[i].to(secondary_dev, non_blocking=False)
                mm.soft_empty_cache()

            # srijithr forward_pre_hook на блоке split_idx —
            # двигает входной hidden_states с primary_dev на secondary_dev
            # ДО forward на этом блоке. Не ломает autograd graph,
            # потому что hooks возвращает contract-modified inputs,
            # а не сам output.
            _remove_stored_hooks(patcher)
            handle = _install_cross_device_hook(
                blocks[split_idx], primary_dev, secondary_dev
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

    # Meta-флаг для downstream-нод: «split применён»
    try:
        patcher.model_options["ltx2_multigpu_split"] = {
            "strategy": strategy,
            "primary": str(primary_dev),
            "secondary": str(secondary_dev),
            "block_split_index": splits[0] if splits else None,
        }
    except Exception:  # noqa: BLE001
        pass

    if verbose:
        try:
            for i, d in enumerate([primary_dev, secondary_dev]):
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info(int(d.index) if d.type == "cuda" else 0)
                    print(
                        f"[ComfyUI-LTX2-MultiGPU] cuda:{int(d.index) if d.type=='cuda' else '?'}"
                        f" free={free/1024**3:.2f} GB total={total/1024**3:.2f} GB"
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


def _move_attrs_by_prefix(parent: Any, prefix: str, device: Any) -> None:
    """Двигает на `device` только те direct-child'ы parent, чей module path
    заканчивается на хвост prefix (последние 2 сегмента).

    Пример:
      prefix = "model.diffusion_model.time_embed."
      tail = "time_embed."
      → двигаются только children, чьё name содержит "time_embed." как suffix
        в своём полном префиксе (через _walk_with_qualname).
    """
    # Последний токен prefix ("time_embed." → "time_embed.") используем как marker
    tail = prefix.rstrip(".").split(".")[-1] if "." in prefix else prefix
    for full_qualname, child in _walk_with_qualname(parent, "", ignore_top=False):
        if not full_qualname:
            continue
        # Match если какой-то подсегмент равен tail
        if any(seg == tail for seg in full_qualname.split(".")):
            try:
                child.to(device, non_blocking=False)
            except Exception:  # noqa: BLE001
                pass


def _walk_with_qualname(parent: Any, _prefix: str, ignore_top: bool = False):
    """Yeld (qualname, module) для всех descendants. qualname — это
    полный dotted-путь относительно root (например 'time_embed.linear_1')."""
    for child_name, child in parent.named_children():
        qn = f"{_prefix}{child_name}" if _prefix else child_name
        yield qn, child
        yield from _walk_with_qualname(child, qn + ".", False)


def load_gemma_hybrid(
    encoder_name: str, projection_name: str, verbose: bool = False
) -> Any:
    """Gemma 12B FP4 → cuda:1, text_projection → cuda:0.

    ⚠️ Phase 3 stub. Вернуть правильный `comfy.sd.CLIP`-совместимый объект можно
    только через GCP/Loaders, которые умеют строить Gemma3Adapter — а они пока
    либо экспериментальные (`kkjais/ComfyUI-Gemma3`), либо недоступны в общем
    ComfyUI workflow. Это честный NotImplementedError с roadmap.

    Roadmap для Phase 3:
      1. Прочитать safetensors Gemma 3 12B FP4 → state_dict.
      2. Прочитать safetensors text_projection → state_dict.
      3. Через `comfy.text_encoder_loader.GEMMA3` или kkjais' adapter
         построить nn.Module (encoder@cuda:1, projection@cuda:0).
      4. Обёрнуть в ModelPatcher (R3).
      5. Вернуть patcher — sampler подхватит естественно.
    """
    raise NotImplementedError(
        "load_gemma_hybrid — Phase 3 task. См. docstring для roadmap. "
        "Phase 2 покрывает только DiT split. Gemma adapter — отдельная задача."
    )

    if verbose:
        print(
            f"[ComfyUI-LTX2-MultiGPU] gemma={encoder_name} proj={projection_name}"
        )

    primary_dev, secondary_dev = resolve_devices()

    try:
        from comfy import sd as comfy_sd  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("comfy.sd недоступен") from exc

    # ── text encoder (Gemma 12B FP4) → cuda:1 ───────────────────────────────
    enc_path = folder_paths.get_full_path("text_encoders", encoder_name)
    state_enc = comfy_sd.load_state_dict_for_model(enc_path) if hasattr(
        comfy_sd, "load_state_dict_for_model"
    ) else None

    if state_enc is None:
        # Fallback: просто читаем safetensors напрямую → dict[fp32/fp16 tensor]
        from safetensors import safe_open  # type: ignore[import-not-found]
        state_enc = {}
        with safe_open(enc_path, framework="pt", device=str(secondary_dev)) as f:
            for key in f.keys():
                state_enc[key] = f.get_tensor(key).to(secondary_dev)

    # ── text_projection → cuda:0 ─────────────────────────────────────────────
    proj_path = folder_paths.get_full_path("text_encoders", projection_name)
    if hasattr(comfy_sd, "load_state_dict_for_model"):
        state_proj = comfy_sd.load_state_dict_for_model(proj_path)
    else:
        from safetensors import safe_open  # type: ignore[import-not-found]
        state_proj = {}
        with safe_open(proj_path, framework="pt", device=str(primary_dev)) as f:
            for key in f.keys():
                state_proj[key] = f.get_tensor(key).to(primary_dev)

    # Concrete text-encoder module: попробуем вызвать ComfyUI build path,
    # если недоступен — вернём минимальный dict-объект, который compat-layer
    # DualCLIPLoader сможет разобрать.
    try:
        from comfy.sd import CLIPType  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        CLIPType = None  # type: ignore[assignment]

    out: dict[str, Any] = {
        "encoder_state": state_enc,
        "projection_state": state_proj,
        "primary_dev": str(primary_dev),
        "secondary_dev": str(secondary_dev),
        "clip_type": CLIPType.GEMMA if CLIPType is not None else None,
    }
    if verbose:
        print("[ComfyUI-LTX2-MultiGPU] GemmaHybrid ready: encoder@cuda:1, proj@cuda:0")
    return out


def apply_strategy(patcher: Any, strategy: str) -> Any:
    """Применяет новую стратегию к уже загруженному ModelPatcher.

    NB: внутри использует cache модель — повторное перемещение блоков между
    GPU. Удаляет старые forward_hook'и через .modules() и пересоздаёт.
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
    splits = _split_blocks_indices(strategy)
    is_pipeline = strategy == "pipeline"
    is_single0 = strategy == "single_cuda0"
    is_single1 = strategy == "single_cuda1"

    if is_single0:
        target_dev = primary_dev
    elif is_single1 or is_pipeline:
        target_dev = secondary_dev
    else:
        target_dev = None

    if target_dev is not None:
        with mm.cuda_device_context(target_dev):
            inner.to(target_dev, non_blocking=False)
    elif blocks is not None and splits:
        split_idx = splits[0]
        with mm.cuda_device_context(primary_dev):
            for i in range(0, split_idx):
                blocks[i].to(primary_dev)
        with mm.cuda_device_context(secondary_dev):
            for i in range(split_idx, len(blocks)):
                blocks[i].to(secondary_dev)
        _remove_stored_hooks(patcher)
        handle = _install_cross_device_hook(
            blocks[split_idx], primary_dev, secondary_dev
        )
        if handle is not None:
            _store_hook(patcher, handle)

    try:
        patcher.model_options["ltx2_multigpu_split"] = {
            "strategy": strategy,
            "primary": str(primary_dev),
            "secondary": str(secondary_dev),
            "block_split_index": splits[0] if splits else None,
        }
    except Exception:  # noqa: BLE001
        pass
    mm.soft_empty_cache()
    return patcher
