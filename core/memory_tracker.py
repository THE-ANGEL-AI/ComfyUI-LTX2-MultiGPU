"""core/memory_tracker.py — Quant-aware VRAM budget + RAM tracking + auto-strategy.

Kaggle Edition (2026-06-30): учитывает реальные ограничения 2×T4 (14.5 GB VRAM
каждая) и системные 29 GB RAM. Читает GGUF header для точного расчёта
квантованного размера вместо слепого fp16-допущения.

Подход:
  1. ``gguf_quant_aware_bytes(path)`` — читает header, суммирует биты
     по реальным типам квантизации (Q4_K_M ≈ 4.5 bpw, Q5_K_M ≈ 5.5 bpw).
  2. VRAM estimate = квантованный размер (city96 GGMLOps держит веса
     квантованными в VRAM, а НЕ fp16).
  3. RAM tracking: file size (mmap) + Python/ComfyUI overhead vs 29 GB.
  4. Auto-strategy: проверяет все стратегии, выбирает лучшую.
  5. Kaggle-специфичные константы и предупреждения.
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


# ════════════════════════════════════════════════════════════════════════════
# Kaggle execution constraints (T4×2)
# ════════════════════════════════════════════════════════════════════════════
KAGGLE_SYSTEM_RAM_GB = 29.0
KAGGLE_VRAM_PER_T4_GB = 14.5
PYTHON_COMFY_RAM_OVERHEAD_GB = 3.5  # Python + ComfyUI + dependencies
KAGGLE_VRAM_RESERVED_GB = 1.0      # ~1 GB резерва на Kaggle T4 (система/драйверы)

# ════════════════════════════════════════════════════════════════════════════
# Approximate bits-per-element for GGUF quant types
#
# Значения — средние по всем слоям (в реальном GGUF разные тензоры могут
# иметь немного разный bpw в зависимости от важности слоя). Для Unsloth
# Dynamic размер файла тот же что у стандартного imatrix-кванта.
# ════════════════════════════════════════════════════════════════════════════
QUANT_BITS_APPROX: dict[str, float] = {
    "Q2_K": 2.6,
    "Q2_K_S": 2.5,
    "Q2_K_XL": 2.6,
    "Q3_K_S": 3.0,
    "Q3_K_M": 3.3,
    "Q3_K_L": 3.6,
    "Q3_K_XL": 3.5,
    "IQ3_XXS": 3.0,
    "Q4_0": 4.5,
    "Q4_1": 5.0,
    "Q4_K_S": 4.0,
    "Q4_K_M": 4.5,
    "Q4_NL": 4.5,
    "Q5_0": 5.5,
    "Q5_1": 6.0,
    "Q5_K_S": 5.0,
    "Q5_K_M": 5.5,
    "Q6_K": 6.6,
    "Q8_0": 8.5,
    "Q8_1": 8.5,
    "BF16": 16.0,
    "F16": 16.0,
    "F32": 32.0,
}
_DEFAULT_BITS = 16.0  # fallback для неизвестных типов

# Bytes per element в fp16 (для справки — НЕ используется для VRAM prediction).
FP16_BYTES_PER_ELEMENT = 2

# Ориентировочные GB для служебных компонент.
COMPONENT_FOOTPRINT_GB: dict[str, float] = {
    "video_vae": 1.35,
    "audio_vae": 0.35,
    "latent_upscaler": 0.95,
    "text_projection": 2.15,
    "gemma_fp4": 7.5,
    "loras_estimate": 2.55,
    "sage_attention_scratch": 2.1,
}


__all__ = [
    "estimate_vram_budget",
    "gguf_estimate_bytes",
    "gguf_quant_aware_bytes",
    "auto_select_strategy",
    "KAGGLE_SYSTEM_RAM_GB",
    "KAGGLE_VRAM_PER_T4_GB",
    "QUANT_BITS_APPROX",
]


def _mm_soft_empty_cache() -> None:
    if torch is None or not torch.cuda.is_available():
        return
    try:
        from comfy import model_management as mm  # type: ignore[import-not-found]
        mm.soft_empty_cache()
    except Exception:  # noqa: BLE001
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolve_path(name: str, *candidates: str) -> str | None:
    """Ищет файл по имени в candidate-папках через folder_paths."""
    if folder_paths is None:
        return None
    for folder in candidates:
        try:
            p = folder_paths.get_full_path(folder, name)
        except Exception:  # noqa: BLE001
            p = None
        if p:
            return p
    return None


# ════════════════════════════════════════════════════════════════════════════
# Quant-aware size estimation
# ════════════════════════════════════════════════════════════════════════════
def gguf_quant_aware_bytes(gguf_path: str) -> tuple[int, int, int]:
    """Возвращает (file_size_bytes, vram_quant_estimate_bytes, fp16_equivalent_bytes).

    VRAM estimate отражает реальный размер с city96 lazy GGMLOps —
    веса хранятся квантованными в VRAM, а НЕ как fp16.
    fp16_equivalent — справочно (сколько бы заняла полная деквантизация).

    Если файл не найден или header не читается — возвращает (0, 0, 0).
    """
    import os

    # File size (mmap — столько же займёт в RAM при обращении к страницам).
    try:
        file_size = os.path.getsize(gguf_path)
    except OSError:
        file_size = 0

    try:
        from .gguf_reader import read_gguf_header
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "gguf_reader недоступен — запустите в составе пакета ComfyUI-LTX2-MultiGPU"
        ) from exc

    try:
        header = read_gguf_header(gguf_path)
    except Exception:  # noqa: BLE001
        return file_size, 0, 0

    total_bits = 0.0
    fp16_bytes = 0
    quant_tensor_count = 0

    for t in header.get("tensors", []):
        elements = int(t.get("n_elements", 0))
        ttype = str(t.get("tensor_type", "?"))
        bits_per_element = QUANT_BITS_APPROX.get(ttype, _DEFAULT_BITS)
        total_bits += elements * bits_per_element
        fp16_bytes += elements * FP16_BYTES_PER_ELEMENT
        if ttype != "?" and bits_per_element < 16.0:
            quant_tensor_count += 1

    vram_quant_estimate = int(total_bits / 8.0)

    return file_size, vram_quant_estimate, int(fp16_bytes)


def _quant_name_from_path(gguf_path: str) -> str:
    """Извлекает имя квантизации из имени файла (e.g. 'Q4_K_M')."""
    import os
    basename = os.path.basename(gguf_path).upper()
    candidates = [
        "UD-Q4_K_M", "UD-Q5_K_M", "UD-Q3_K_XL", "UD-Q2_K_XL",
        "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S",
        "Q4_K_M", "Q4_K_S", "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S",
        "Q2_K_XL", "Q2_K", "IQ3_XXS",
    ]
    for c in candidates:
        if c in basename:
            return c
    return "?"


# ════════════════════════════════════════════════════════════════════════════
# Legacy fp16 estimator (backward compat)
# ════════════════════════════════════════════════════════════════════════════
def gguf_estimate_bytes(gguf_path: str) -> tuple[int, int]:
    """Возвращает (total_elements, estimated_fp16_bytes) — LEGACY.

    Для новых вызовов используйте ``gguf_quant_aware_bytes()``.
    """
    try:
        from .gguf_reader import read_gguf_header
    except Exception:  # noqa: BLE001
        import importlib
        try:
            mod = importlib.import_module("core.gguf_reader")
            read_gguf_header = mod.read_gguf_header
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "gguf_reader недоступен — нужно запускать в составе пакета"
            ) from exc

    header = read_gguf_header(gguf_path)
    total_elt = int(header.get("total_elements", 0))
    fp16_bytes = total_elt * FP16_BYTES_PER_ELEMENT
    return total_elt, fp16_bytes


# ════════════════════════════════════════════════════════════════════════════
# Per-strategy VRAM projection
# ════════════════════════════════════════════════════════════════════════════
def _project(
    strategy: str,
    dit_gb: float,
    gemma_gb: float,
    components: dict[str, float],
) -> tuple[float, float]:
    """Per-strategy VRAM projection."""
    other_cuda0_gb = (
        components.get("text_projection", 0.0)
        + components.get("video_vae", 0.0)
        + components.get("latent_upscaler", 0.0)
        + components.get("sage_attention_scratch", 0.0)
    )
    other_cuda1_gb = (
        components.get("audio_vae", 0.0)
        + components.get("sage_attention_scratch", 0.0)
    )
    loras_gb = components.get("loras_estimate", 0.0)
    if strategy == "single_cuda0":
        return dit_gb + gemma_gb + other_cuda0_gb + loras_gb, other_cuda1_gb
    if strategy == "single_cuda1":
        return other_cuda0_gb, dit_gb + gemma_gb + other_cuda1_gb + loras_gb
    if strategy == "pipeline":
        return gemma_gb + other_cuda0_gb, dit_gb + other_cuda1_gb + loras_gb
    if strategy == "blocks_50_50":
        return (
            dit_gb / 2 + other_cuda0_gb + loras_gb,
            dit_gb / 2 + gemma_gb + other_cuda1_gb,
        )
    if strategy == "blocks_30_70":
        return (
            dit_gb * 0.3 + other_cuda0_gb + loras_gb,
            dit_gb * 0.7 + gemma_gb + other_cuda1_gb,
        )
    return 0.0, 0.0


# ════════════════════════════════════════════════════════════════════════════
# Auto-strategy selector
# ════════════════════════════════════════════════════════════════════════════
def auto_select_strategy(
    dit_gb: float,
    gemma_gb: float,
    cap0: float,
    cap1: float,
    components: dict[str, float] | None = None,
    virtual_vram_gb: float = 0.0,
) -> str | None:
    """Выбирает лучшую стратегию которая помещается в VRAM.

    Приоритет: blocks_50_50 > blocks_30_70 > pipeline.
    Возвращает None если ни одна стратегия не помещается.

    NEW (v0.6.2-pre, WhitePaper §8.6): ``virtual_vram_gb`` бонус к cap0
    (primary) — reserved VRAM gap для DiT split без реального alloc.
    Clamp [0, 16]; clamp [0, 8] для safety в проекции. Silent default 0.
    """
    comps = components or COMPONENT_FOOTPRINT_GB
    eff_vram = max(0.0, min(8.0, float(virtual_vram_gb)))
    eff_cap0 = cap0 + eff_vram
    for strategy in ("blocks_50_50", "blocks_30_70", "pipeline"):
        c0, c1 = _project(strategy, dit_gb, gemma_gb, comps)
        if c0 <= eff_cap0 and c1 <= cap1:
            return strategy
    return None


# ════════════════════════════════════════════════════════════════════════════
# Main diagnostic function (Kaggle Edition)
# ════════════════════════════════════════════════════════════════════════════
def estimate_vram_budget(
    gguf_name: str,
    gemma_name: str = "",
    purge_cache: bool = True,
    virtual_vram_gb: float = 0.0,
) -> str:
    """Подсчитывает ожидаемые VRAM-затраты + рекомендация стратегии.

    Kaggle Edition:
      - Читает реальный GGUF quant type (не fp16 допущение)
      - RAM бюджет: file size + overhead vs 29 GB
      - VRAM caps: реальные 14.5 GB на T4
      - Auto-select стратегии
      - Предупреждения для неподходящих квантов
    """
    if purge_cache:
        _mm_soft_empty_cache()

    lines: list[str] = [
        "LTX-2 MultiGPU Memory Diagnostics (Kaggle Edition)",
        "─" * 55,
    ]

    # ── DiT: quant-aware размер ──────────────────────────────────────────
    dit_vram_gb = 0.0
    dit_fp16_gb = 0.0
    dit_file_gb = 0.0
    quant_name = "?"
    if gguf_name:
        dit_path = _resolve_path(
            gguf_name, "diffusion_models", "checkpoints", "unet"
        )
        if dit_path:
            try:
                file_bytes, vram_bytes, fp16_bytes = gguf_quant_aware_bytes(dit_path)
                dit_file_gb = file_bytes / 1024**3
                dit_vram_gb = vram_bytes / 1024**3
                dit_fp16_gb = fp16_bytes / 1024**3
                quant_name = _quant_name_from_path(dit_path)
                lines.append(
                    f"DiT  {gguf_name}"
                )
                lines.append(
                    f"     Quant: {quant_name}, "
                    f"file={dit_file_gb:.2f} GB, "
                    f"VRAM≈{dit_vram_gb:.2f} GB (квант), "
                    f"fp16≈{dit_fp16_gb:.2f} GB (справочно)"
                )
            except Exception as exc:  # noqa: BLE001
                lines.append(f"DiT  {gguf_name}: header read FAILED: {exc}")
        else:
            lines.append(
                f"DiT  {gguf_name}: НЕ НАЙДЕН в diffusion_models/checkpoints/unet"
            )

    # ── Gemma ────────────────────────────────────────────────────────────
    gemma_gb = COMPONENT_FOOTPRINT_GB["gemma_fp4"]
    if gemma_name:
        lines.append(f"Gemma {gemma_name}: ~{gemma_gb:.2f} GB (FP4 + KV-cache)")
    else:
        lines.append(f"Gemma (default): ~{gemma_gb:.2f} GB (FP4 + KV-cache)")

    # ── System RAM budget ────────────────────────────────────────────────
    lines.append("")
    lines.append("─ RAM Budget ─")
    ram_used = dit_file_gb + PYTHON_COMFY_RAM_OVERHEAD_GB
    lines.append(
        f"  DiT mmap: {dit_file_gb:.2f} GB + "
        f"overhead: {PYTHON_COMFY_RAM_OVERHEAD_GB:.2f} GB "
        f"= {ram_used:.2f} GB / {KAGGLE_SYSTEM_RAM_GB:.1f} GB RAM"
    )
    if ram_used > KAGGLE_SYSTEM_RAM_GB:
        lines.append(
            f"  ⚠️  OOM RISK: {ram_used:.2f} GB > {KAGGLE_SYSTEM_RAM_GB:.1f} GB RAM! "
            f"mmap будет thrash'ить. Используйте меньший квант."
        )
    else:
        lines.append(f"  ✓  RAM OK ({KAGGLE_SYSTEM_RAM_GB - ram_used:.1f} GB свободно)")

    # ── VRAM per card ────────────────────────────────────────────────────
    lines.append("")
    lines.append("─ VRAM Budget per Strategy ─")
    free_per_card: list[tuple[int, float, float]] = []
    if torch is not None and torch.cuda.is_available():
        for i in range(int(torch.cuda.device_count())):
            free, total = torch.cuda.mem_get_info(int(i))
            free_per_card.append((i, free / 1024**3, total / 1024**3))

    # Kaggle-aware caps: используем свободную VRAM (после purge_cache).
    # Fallback — тоже свободная оценка (total − зарезервировано).
    cap0 = (
        free_per_card[0][1] if len(free_per_card) >= 1
        else KAGGLE_VRAM_PER_T4_GB - KAGGLE_VRAM_RESERVED_GB
    )
    cap1 = (
        free_per_card[1][1] if len(free_per_card) >= 2
        else cap0
    )

    # NEW (v0.6.2-pre, WhitePaper §8.6): virtual-aware cap boost (cuda:0).
    eff_vram = max(0.0, min(8.0, float(virtual_vram_gb)))
    eff_cap0 = cap0 + eff_vram
    if eff_vram > 0:
        lines.append(
            f"  cuda:0 cap = {cap0:.2f} GB + virtual_vram_gb={eff_vram:.1f} GB "
            f"= {eff_cap0:.2f} GB effective"
        )
    else:
        lines.append(f"  cuda:0 cap = {cap0:.2f} GB  |  cuda:1 cap = {cap1:.2f} GB")
    lines.append("")

    valid_strategies: list[str] = []
    for strategy in ("blocks_50_50", "blocks_30_70", "pipeline"):
        c0, c1 = _project(strategy, dit_vram_gb, gemma_gb, COMPONENT_FOOTPRINT_GB)
        ok0 = "✓" if c0 <= eff_cap0 else "✗ OOM"
        ok1 = "✓" if c1 <= cap1 else "✗ OOM"
        lines.append(
            f"  {strategy:>14s}: cuda0={c0:5.2f} GB {ok0:6s} | "
            f"cuda1={c1:5.2f} GB {ok1:6s}"
        )
        if c0 <= eff_cap0 and c1 <= cap1:
            valid_strategies.append(strategy)

    # ── Текущее состояние карт ────────────────────────────────────────────
    lines.append("")
    lines.append("─ Current VRAM State ─")
    if free_per_card:
        for i, free, total in free_per_card:
            used = total - free
            lines.append(
                f"  cuda:{i} used={used:.2f} GB / total={total:.2f} GB "
                f"(free={free:.2f} GB)"
            )
    else:
        lines.append("  CUDA недоступна (тестовый режим)")

    # ── Auto-select + рекомендация ───────────────────────────────────────
    lines.append("")
    lines.append("─ Recommendation ─")
    best = auto_select_strategy(
        dit_vram_gb, gemma_gb, cap0, cap1,
        COMPONENT_FOOTPRINT_GB, virtual_vram_gb=virtual_vram_gb,
    )
    if best:
        lines.append(f"✅ AUTO: {best}")
    else:
        lines.append("🚨 НИ ОДНА стратегия не помещается в VRAM!")
        if dit_vram_gb > 6.0:
            lines.append(
                f"💡 Слишком большой квант ({quant_name}). "
                f"Попробуйте Q4_K_M (~14 GB file) или Q3_K_XL (~12 GB file)."
            )
        if dit_vram_gb > 3.0 and "blocks_30_70" not in valid_strategies:
            lines.append(
                "💡 Попробуйте blocks_30_70 (30% DiT на cuda:0, 70% на cuda:1) "
                "или pipeline (весь DiT на cuda:1, Gemma на cuda:0)."
            )

    return "\n".join(lines)
