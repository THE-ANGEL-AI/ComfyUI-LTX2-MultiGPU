"""core/vram_parking.py — VRAM parking for DiT between sampling passes.

White Paper §8.1 CRITICAL: between Pass 1 (KSampler) and Pass 2 (KSampler),
ComfyUI runs VAE decode → upscale → VAE encode. DiT blocks on cuda:0 (~9 GB)
compete with VAE/upscaler → OOM. This module provides:

  ``park_dit(patcher)``   — moves ALL DiT blocks + embed/head to CPU
  ``unpark_dit(patcher)`` — restores blocks to original GPU layout

Unpark delegates to ``apply_strategy`` (core/gguf_split.py), which:
  1. Removes old hooks
  2. Resolves devices
  3. Moves blocks to GPUs
  4. Re-installs cross-device forward hook
  5. Moves embed/head layers
  6. Re-locks inner.to() (Risk #7)
  7. Calls soft_empty_cache()

All idempotent via ``patcher.model_options["ltx2_multigpu_split"]["parked"]``.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]


__all__ = ["park_dit", "unpark_dit"]


def _is_parked(patcher: Any) -> bool:
    """Проверяет флаг parked в model_options."""
    try:
        return bool(
            patcher.model_options.get("ltx2_multigpu_split", {}).get("parked", False)
        )
    except Exception:  # noqa: BLE001
        return False


def _set_parked(patcher: Any, state: bool) -> None:
    """Устанавливает флаг parked в model_options."""
    try:
        opts = patcher.model_options.setdefault("ltx2_multigpu_split", {})
        opts["parked"] = state
    except Exception:  # noqa: BLE001
        pass


def park_dit(patcher: Any) -> Any:
    """Перемещает ВСЕ блоки DiT + embed/head на CPU, освобождая VRAM.

    Идемпотентен: если уже parked — сразу возвращает patcher.
    Если нет ltx2_multigpu_split в model_options — WARN + no-op.

    Безопасен для GGUF: использует per-block ``.to('cpu')`` (module-level),
    а не ``inner.to('cpu')`` (top-level) — это НЕ триггерит dequant через
    city96 GGMLOps, потому что block.to() идёт через nn.Module native метод.
    """
    if torch is None:
        return patcher

    opts = patcher.model_options.get("ltx2_multigpu_split") if hasattr(patcher, "model_options") else None
    if not opts:
        print(
            "[ComfyUI-LTX2-MultiGPU] WARN: park_dit: нет ltx2_multigpu_split в model_options — "
            "парковка невозможна (это DiT модель от HybridSplitLoader?)"
        )
        return patcher

    if _is_parked(patcher):
        return patcher

    try:
        from comfy import model_management as mm  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return patcher

    from .gguf_split import _remove_stored_hooks, _move_modules_with_prefix
    from .gguf_split import EMBED_AND_HEAD_REL, LTX2_DIT_BLOCK_COUNT

    inner = patcher.model
    diffusion = getattr(inner, "diffusion_model", inner)
    blocks = getattr(diffusion, "transformer_blocks", None)
    cpu_dev = torch.device("cpu")

    # Убираем активные cross-device forward-хуки.
    _remove_stored_hooks(patcher)

    # CRITICAL (v0.6.0-pre fix): _lock_inner_to_recursive (Round 3 / BUG-6) после
    # hybrid_split_gguf навешивает _no_op_to_patch на КАЖДЫЙ submodule (включая
    # каждый transformer_block). Без unlock блоки DiT silent no-op'нули бы
    # при block.to(cpu). Waiting на sampler между KSampler-step не помогает —
    # это явно user-action node, не sampler callback.
    # Паттерн: unlock → move → re-lock (для sampler safety после парковки,
    # если unpark не сработает — DiT остаётся запертым от sampler'а на CPU,
    # вместо silent migrate на cuda:0 mid-sampling).
    from .gguf_split import _unlock_inner_to_recursive, _lock_inner_to_recursive
    _unlock_inner_to_recursive(inner)

    # Перемещаем все блоки DiT на CPU (per-block, не inner.to('cpu')).
    # module-level .to() работает напрямую с nn.Module и не проходит
    # через запертый _lock_inner_to патч на top-level.
    if blocks is not None and len(blocks) == LTX2_DIT_BLOCK_COUNT:
        for block in blocks:
            try:
                block.to(cpu_dev, non_blocking=False)
            except Exception:  # noqa: BLE001
                pass
    else:
        # Fallback: если transformer_blocks не найдены — двигаем весь inner.
        # Используем _ltx2_original_to (реальный .to()) — запертый inner.to
        # это no-op для device-переносов после _lock_inner_to.
        try:
            _original_to = getattr(inner, "_ltx2_original_to", inner.to)
            _original_to(cpu_dev, non_blocking=False)
        except Exception:  # noqa: BLE001
            pass

    # Embed/head слои тоже на CPU.
    _move_modules_with_prefix(diffusion, cpu_dev, *EMBED_AND_HEAD_REL)

    # Re-lock: теперь когда DiT на CPU, sampler не должен мочь его
    # перетащить обратно на cuda:0 mid-VAE-decode (что занимает 5-10 сек
    # и DiT не используется в этом диапазоне, но ComfyUI может попытаться
    # blanket move). Тем более что device-cpu теперь — fallback на
    # primary_gen = mm.get_torch_device() — без lock может мигрировать.
    _lock_inner_to_recursive(inner)

    mm.soft_empty_cache()
    _set_parked(patcher, True)

    strategy = opts.get("strategy", "?")
    primary = opts.get("primary", "?")
    effective_donor = opts.get("effective_donor", "?")
    print(
        f"[ComfyUI-LTX2-MultiGPU] park_dit: DiT -> CPU "
        f"(strategy={strategy}, primary={primary}, donor={effective_donor})"
    )

    return patcher


def unpark_dit(patcher: Any) -> Any:
    """Восстанавливает DiT блоки на GPU в исходном layout.

    Идемпотентен: если НЕ parked — сразу возвращает patcher.
    Делегирует в ``apply_strategy`` для точного восстановления split.

    ``apply_strategy`` делает:
      1. Очищает старые хуки
      2. Разрешает устройства (primary / donor)
      3. Перемещает блоки на GPU согласно strategy
      4. Устанавливает cross-device forward-hook
      5. Переносит embed/head слои на primary
      6. Перезапирает inner.to() (Risk #7)
      7. Вызывает soft_empty_cache()
    """
    if torch is None:
        return patcher

    opts = patcher.model_options.get("ltx2_multigpu_split") if hasattr(patcher, "model_options") else None
    if not opts:
        return patcher

    if not _is_parked(patcher):
        return patcher

    from .gguf_split import apply_strategy

    # Читаем значения ДО вызова apply_strategy — она заменит dict целиком,
    # и opts-ссылка станет stale.
    strategy = opts.get("strategy", "blocks_50_50")
    donor_spec = opts.get("donor_spec", "auto")
    primary = opts.get("primary", "?")
    effective_donor = opts.get("effective_donor", "?")

    # apply_strategy восстанавливает полный GPU layout.
    apply_strategy(patcher, strategy=strategy, donor_device=donor_spec)

    _set_parked(patcher, False)

    print(
        f"[ComfyUI-LTX2-MultiGPU] unpark_dit: DiT restored to GPU "
        f"(strategy={strategy}, primary={primary}, donor={effective_donor})"
    )

    return patcher
