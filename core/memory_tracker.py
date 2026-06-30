"""core/memory_tracker.py — Phase 2 реализация: real VRAM budget.

Подход:
  1. GGUF-файл читается через `gguf.GGUFReader(path)` в режиме mmap.
     Это даёт доступ к tensor names, shapes, quantization types
     БЕЗ загрузки самих весов в RAM/VRAM.
  2. fp16-эквивалент каждого тензора = n_elements * 2 байта.
  3. Суммы по группам: DiT blocks / embed / head / Gemma / VAE.
  4. Сравнение с реальным VRAM через torch.cuda.mem_get_info().
  5. Рекомендация стратегии (50/50 / 30/70 / pipeline).
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


# Bytes per element в fp16 (стандарт для наших split-целей).
FP16_BYTES_PER_ELEMENT = 2

# Сколько байт дают разные GGUF-квантизации в fp16 (после dequant).
# Это грубая "средняя" оценка для VRAM после split + dequant.
APPROX_FP16_BYTES_PER_GGUF_BYTES = 2.0  # Q5_K_M ≈ 5.5 бит → после dequant ≈ 16 бит

# Ориентировочные bytes для служебных компонент (не измеряем — фиксируем).
COMPONENT_FOOTPRINT_GB: dict[str, float] = {
    "video_vae": 1.35,
    "audio_vae": 0.35,
    "latent_upscaler": 0.95,
    "text_projection": 2.15,
    "gemma_fp4": 7.5,           # + 1.5 GB KV-cache headroom
    "loras_estimate": 2.55,
    "sage_attention_scratch": 2.1,
}


__all__ = ["estimate_vram_budget", "gguf_estimate_bytes"]


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


def gguf_estimate_bytes(gguf_path: str) -> tuple[int, int]:
    """Возвращает (total_elements, estimated_fp16_bytes) для GGUF-файла.

    Header-только через gguf.GGUFReader (mmap). Никогда не загружает веса.
    """
    try:
        from core.gguf_reader import read_gguf_header  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        # Если относительный импорт сломан (запуск вне пакета) — fallback
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


def estimate_vram_budget(
    gguf_name: str, gemma_name: str, purge_cache: bool = True
) -> str:
    """Подсчитывает ожидаемые VRAM-затраты + рекомендация стратегии.

    Поля отчёта:
      - DiT estimate (fp16)
      - Per-card projection (cuda:0 / cuda:1) для каждой стратегии
      - Текущий free VRAM на каждой карте
      - "VRAM OK" / "OOM, fallback to pipeline"
    """
    if purge_cache:
        _mm_soft_empty_cache()

    lines: list[str] = ["LTX-2 MultiGPU Memory Diagnostics (Phase 2)"]

    # ── Диагностика GGUF ────────────────────────────────────────────────────
    dit_fp16_bytes = 0
    if gguf_name:
        dit_path = _resolve_path(gguf_name, "diffusion_models", "checkpoints", "unet")
        if dit_path:
            try:
                total_elt, dit_fp16_bytes = gguf_estimate_bytes(dit_path)
                lines.append(
                    f"DiT {gguf_name}: {len(gguf_name) and total_elt:,} elements → "
                    f"{dit_fp16_bytes / 1024**3:.2f} GB fp16 (from GGUF header)"
                )
            except Exception as exc:  # noqa: BLE001
                lines.append(f"DiT {gguf_name}: header read FAILED: {exc}")
        else:
            lines.append(f"DiT {gguf_name}: NOT FOUND in diffusion_models/checkpoints/unet")
    dit_gb = dit_fp16_bytes / 1024**3

    # ── Gemma, VAE и прочие ─────────────────────────────────────────────────
    lines.append(f"Gemma {gemma_name}: ~{COMPONENT_FOOTPRINT_GB['gemma_fp4']:.2f} GB (FP4 + KV-cache est)")
    gemma_gb = COMPONENT_FOOTPRINT_GB["gemma_fp4"]
    other_cuda0_gb = (
        COMPONENT_FOOTPRINT_GB["text_projection"]
        + COMPONENT_FOOTPRINT_GB["video_vae"]
        + COMPONENT_FOOTPRINT_GB["latent_upscaler"]
        + COMPONENT_FOOTPRINT_GB["sage_attention_scratch"]
    )
    other_cuda1_gb = (
        COMPONENT_FOOTPRINT_GB["audio_vae"]
        + COMPONENT_FOOTPRINT_GB["sage_attention_scratch"]
    )
    loras_gb = COMPONENT_FOOTPRINT_GB["loras_estimate"]

    # ── Прогноз для каждой стратегии ─────────────────────────────────────────
    def _project(strategy: str) -> tuple[float, float]:
        """Возвращает (cuda0_gb_load, cuda1_gb_load) для стратегии.

        В v0.2.1 pipeline-strategy тоже включает ``loras_estimate`` (раньше был
        silent-bug: LoRы мерджатся **перед** KSampler-loop в любой стратегии,
        включая 'pipeline', но _project для pipeline их не учитывал → false-OK
        projection → real OOM на user-machine с LoRA workload).

        Принцип: где бы ни лежал DiT (primary или donor), туда и мерджится
        LoRA перед sampling-loop. Поэтому:
        - cuda:0 содержит DiT (single_cuda0 / blocks_* 0-share): +loras_gb
        - cuda:1 содержит DiT (single_cuda1 / pipeline / blocks_* 1-share): +loras_gb
        """
        if strategy == "single_cuda0":
            return dit_gb + gemma_gb + other_cuda0_gb + loras_gb, other_cuda1_gb
        if strategy == "single_cuda1":
            return other_cuda0_gb, dit_gb + gemma_gb + other_cuda1_gb + loras_gb
        if strategy == "pipeline":
            # DiT целиком на cuda:1; Gemma + LoRы (мерджатся в DiT @ cuda:1) на cuda:1;
            # Gemma занимает cuda:0 для encoder (text conditioning).
            return gemma_gb + other_cuda0_gb, dit_gb + other_cuda1_gb + loras_gb
        if strategy == "blocks_50_50":
            cuda0 = dit_gb / 2 + other_cuda0_gb + loras_gb
            cuda1 = dit_gb / 2 + gemma_gb + other_cuda1_gb
            return cuda0, cuda1
        if strategy == "blocks_30_70":
            cuda0 = dit_gb * 0.3 + other_cuda0_gb + loras_gb
            cuda1 = dit_gb * 0.7 + gemma_gb + other_cuda1_gb
            return cuda0, cuda1
        return 0.0, 0.0

    lines.append("")
    lines.append("Стратегии: cuda:0_load / cuda:1_load vs доступно")
    free_per_card: list[tuple[int, float, float]] = []
    if torch is not None and torch.cuda.is_available():
        for i in range(int(torch.cuda.device_count())):
            free, total = torch.cuda.mem_get_info(int(i))
            free_per_card.append((i, free / 1024**3, total / 1024**3))
    cap0 = free_per_card[0][2] if len(free_per_card) >= 1 else 15.0
    cap1 = free_per_card[1][2] if len(free_per_card) >= 2 else cap0
    for strategy in ("blocks_50_50", "blocks_30_70", "pipeline", "single_cuda0"):
        c0, c1 = _project(strategy)
        ok0 = "✓" if c0 <= cap0 else "✗"
        ok1 = "✓" if c1 <= cap1 else "✗"
        lines.append(
            f"  {strategy:>14s}: cuda0={c0:5.2f} GB {ok0} (cap {cap0:.2f}) | "
            f"cuda1={c1:5.2f} GB {ok1} (cap {cap1:.2f})"
        )

    # ── Реальное состояние карт ─────────────────────────────────────────────
    lines.append("")
    lines.append("Текущая нагрузка:")
    if free_per_card:
        for i, free, total in free_per_card:
            used = total - free
            lines.append(
                f"  cuda:{i} used={used/1024**3:5.2f} GB / total={total/1024**3:5.2f} GB (free={free/1024**3:5.2f})"
            )
    else:
        lines.append("  CUDA недоступна")

    # ── Рекомендация ────────────────────────────────────────────────────────
    lines.append("")
    c0_b, c1_b = _project("blocks_50_50")
    if c0_b <= cap0 and c1_b <= cap1:
        lines.append("РЕКОМЕНДАЦИЯ: blocks_50_50 (DiT 22B разделён пополам)")
    elif _project("blocks_30_70")[1] <= cap1:
        lines.append("РЕКОМЕНДАЦИЯ: blocks_30_70 (меньше DiT на дальней карте)")
    elif _project("pipeline")[0] <= cap0 and _project("pipeline")[1] <= cap1:
        lines.append("РЕКОМЕНДАЦИЯ: pipeline (DiT@cuda:1 + Gemma@cuda:0)")
    else:
        lines.append(
            "ВНИМАНИЕ: ни одна стратегия не помещается → уменьшить разрешение "
            "или использовать quantized более агрессивно (Q4_K_M)."
        )

    return "\n".join(lines)


__all__ = ["estimate_vram_budget", "gguf_estimate_bytes"]
