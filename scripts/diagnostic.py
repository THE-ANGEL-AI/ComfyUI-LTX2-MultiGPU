#!/usr/bin/env python3
"""Standalone pre-flight VRAM diagnostic + nvidia-smi monitor for LTX 2.3.

Used to verify — BEFORE actually loading any weights into VRAM — that a given
GGUF / safetensors combo will fit on the available GPUs under the chosen
split strategy. Mirrors PLAN §5.1 / MODEL_FACTS §2 verbatim in its output
shape and footprint constants.

NEVER imports the custom_nodes package: deliberately standalone so it can
run in any venv / Kaggle notebook kernel without ComfyUI's runtime.

Usage:
    python scripts/diagnostic.py [options]

Examples:
    python scripts/diagnostic.py
    python scripts/diagnostic.py --unet lt2-3-q6_K.gguf --gemma gemma-3-12b-fp4.safetensors
    python scripts/diagnostic.py --strategy pipeline
    python scripts/diagnostic.py --smi-poll-ms 500 --smi-duration-s 60
    python scripts/diagnostic.py --json > diag.json
    python scripts/diagnostic.py --list

Exit codes:
    0   everything fits the requested strategy (or --json emission only)
    1   at least one strategy over budget
    2   nvidia-smi AND torch.cuda both unavailable; using MODEL_FACTS defaults
    3   bad CLI inputs (file not found, strategie unknown, etc.)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

# ─── Heavy deps guarded per R6 ────────────────────────────────────────────
try:
    import gguf  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    gguf = None  # type: ignore[assignment]

try:
    import safetensors  # type: ignore[import-not-found]
    from safetensors import safe_open  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    safetensors = None  # type: ignore[assignment]
    safe_open = None  # type: ignore[assignment]


# ════════════════════════════════════════════════════════════════════════════
# Approximate bits-per-element for GGUF quant types
#
# Синхронизировано с core/memory_tracker.QUANT_BITS_APPROX.
# При изменении там — обновить эту копию.
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

# ════════════════════════════════════════════════════════════════════════════
# Kaggle execution constraints (синхронизировано с core/memory_tracker)
# ════════════════════════════════════════════════════════════════════════════
KAGGLE_SYSTEM_RAM_GB = 29.0
KAGGLE_VRAM_PER_T4_GB = 14.5
KAGGLE_VRAM_RESERVED_GB = 1.0
PYTHON_COMFY_RAM_OVERHEAD_GB = 3.5


# ─── Constants (synced with MODEL_FACTS §2 / PLAN §5.1) ───────────────────
STRATEGIES: tuple[str, ...] = (
    "blocks_50_50",  # default: DiT 0..21 -> cuda:0; 22..43 -> cuda:1
    "blocks_30_70",  # DiT 0..13  -> cuda:0; 14..43 -> cuda:1
    "pipeline",      # DiT на cuda:1 целиком, Gemma на cuda:0
    "single_cuda0",  # всё на cuda:0
    "single_cuda1",  # всё на cuda:1
)

# Bytes-GiB constant (1 GiB = 1024^3 B).
_GIB = 1024 ** 3

# Per-component footprint estimates (GB). Источник: MODEL_FACTS §2.
#
# Ключ `sage_attention_scratch` есть canonical-name, синхронизированный с
# ``core/memory_tracker.COMPONENT_FOOTPRINT_GB``. Локальная копия для standalone-CLI
# использования (diagnostic.py запускается вне ComfyUI, не имеет доступа к
# пакету core/). При изменении ключа в core/memory_tracker.py — синхронизировать
# эту копию (плюс добавить regression-test в tests/test_memory_tracker.py —
# см. test_project_strategy_keys_unified).
COMPONENT_FOOTPRINT_GB: dict[str, float] = {
    "video_vae":              1.35,
    "audio_vae":              0.35,
    "latent_upscaler":        0.95,
    "text_projection":        2.15,
    "gemma_fp4":              7.5,
    "loras_estimate":         2.55,
    "sage_attention_scratch": 2.1,
}


# ─── File-system helpers ─────────────────────────────────────────────────
def _autodetect_project_root(arg: str) -> Path | None:
    """Ищет ComfyUI root (содержит models/diffusion_models/)."""
    if arg:
        p = Path(arg).expanduser().resolve()
        if p.exists():
            return p
    for c in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
        if (c / "models" / "diffusion_models").exists():
            return c
    return None


def _resolve_gguf(spec: str, root: Path | None) -> Path | None:
    if spec:
        p = Path(spec).expanduser()
        if p.is_file():
            return p.resolve()
        if root is not None:
            cand = root / "models" / "diffusion_models" / spec
            if cand.is_file():
                return cand.resolve()
        return None
    if root is None:
        return None
    matches = sorted((root / "models" / "diffusion_models").glob("*.gguf"))
    return matches[0].resolve() if matches else None


def _resolve_safetensors(spec: str, root: Path | None) -> Path | None:
    if spec:
        p = Path(spec).expanduser()
        if p.is_file():
            return p.resolve()
        if root is not None:
            for sub in ("text_encoders", "clip"):
                cand = root / "models" / sub / spec
                if cand.is_file():
                    return cand.resolve()
        return None
    if root is None:
        return None
    for sub in ("text_encoders", "clip"):
        matches = sorted((root / "models" / sub).glob("*.safetensors"))
        if matches:
            return matches[0].resolve()
    return None


def _detect_quant_name(basename: str) -> str:
    """Извлекает имя квантизации из имени файла (e.g. 'Q4_K_M')."""
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


# ─── Header-only file scanners ───────────────────────────────────────────
@dataclass
class GGUFScan:
    path: str
    file_gb: float
    n_tensors: int
    total_elements: int
    fp16_estimate_gb: float       # справочно (полная деквантизация)
    quant_vram_estimate_gb: float  # реальный VRAM с квантованием
    quant_name: str                # имя кванта (e.g. "Q4_K_M")
    fp4_estimate_gb: float | None  # если похоже на упакованный FP4
    
    @classmethod
    def scan(cls, path: Path) -> "GGUFScan":
        size_b = int(path.stat().st_size)
        size_gb = size_b / _GIB
        basename = path.name.upper()
        if gguf is None:
            return cls(
                path=str(path), file_gb=size_gb, n_tensors=0,
                total_elements=0, fp16_estimate_gb=size_gb * 2.0,
                quant_vram_estimate_gb=size_gb,
                quant_name=_detect_quant_name(basename),
                fp4_estimate_gb=None,
            )
        # gguf.GGUFReader() в режиме mmap — header прочитан без загрузки весов
        reader = gguf.GGUFReader(str(path))
        n_elem = sum(int(t.n_elements) for t in reader.tensors)
        fp16_gb = n_elem * 2 / _GIB  # n_elements × 2 байта (fp16)

        # ── Quant-aware размер (Kaggle Edition) ──────────────────────────
        # Итерируем тензоры, смотрим tensor_type и суммируем реальные биты.
        # city96 GGMLOps держит веса квантованными в VRAM — не fp16.
        total_bits = 0.0
        for t in reader.tensors:
            elements = int(t.n_elements)
            ttype = str(getattr(t, "tensor_type", "?"))
            bits_per_element = QUANT_BITS_APPROX.get(ttype, _DEFAULT_BITS)
            total_bits += elements * bits_per_element
        quant_vram_gb = (total_bits / 8.0) / _GIB if n_elem else size_gb
        quant_name = _detect_quant_name(basename)

        # Эвристика «это GGUF Q5/Q6 quantization»:
        # packed bytes per element < 1.0 → значит это quantized, не raw fp16
        bpe_packed = (size_b / max(n_elem, 1)) if n_elem else 0.0
        fp4_gb = (n_elem * 0.5 / _GIB) if (bpe_packed and bpe_packed < 1.0) else None
        return cls(
            path=str(path), file_gb=size_gb,
            n_tensors=len(reader.tensors),
            total_elements=n_elem,
            fp16_estimate_gb=fp16_gb,
            quant_vram_estimate_gb=quant_vram_gb,
            quant_name=quant_name,
            fp4_estimate_gb=fp4_gb,
        )


@dataclass
class SafetensorsScan:
    path: str
    file_gb: float
    n_tensors: int
    total_elements: int
    fp16_estimate_gb: float
    
    @classmethod
    def scan(cls, path: Path) -> "SafetensorsScan":
        size_b = int(path.stat().st_size)
        size_gb = size_b / _GIB
        if safetensors is None or safe_open is None:
            return cls(
                path=str(path), file_gb=size_gb, n_tensors=0,
                total_elements=0, fp16_estimate_gb=size_gb,
            )
        n_elem = 0
        n_t = 0
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                n_elem += int(f.get_tensor(key).numel())
                n_t += 1
        # Если safetensors FP4 (как Gemma 3 12B) — после dequant в fp16:
        #   n_elem * 0.5 байт → n_elem * 2 байт. Heuristic: file size
        #   значительно меньше fp16-объёма (size < 1.5 × n_elem).
        # NB: прежняя формула «size_b * 4 < n_elem» была инвертирована
        # и НЕ срабатывала для FP4 (review B1).
        if size_b and n_elem and (size_b * 1.5 < n_elem):
            fp16_gb = n_elem * 2 / _GIB
        else:
            # bf16/fp16 safetensors — file size ≈ n_elem × 2 bytes
            fp16_gb = size_gb
        return cls(
            path=str(path), file_gb=size_gb,
            n_tensors=n_t, total_elements=n_elem,
            fp16_estimate_gb=fp16_gb,
        )


# ─── nvidia-smi probe ────────────────────────────────────────────────────
def _nvidia_smi_query() -> list[dict[str, str]] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        rows: list[dict[str, str]] = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            rows.append({
                "index":          parts[0],
                "name":           parts[1],
                "memory_total_mib": parts[2],
                "memory_used_mib":  parts[3],
                "memory_free_mib":  parts[4],
                "gpu_util_pct":     parts[5],
            })
        return rows
    except Exception:  # noqa: BLE001
        return None


class _SmiMonitor(threading.Thread):
    """Background ``nvidia-smi`` poller. Daemon — умирает с main process."""
    
    daemon = True
    
    def __init__(
        self,
        interval_ms: int,
        duration_s: int,
        on_sample: Callable[[list[dict[str, str]], float], None],
    ) -> None:
        super().__init__(name="smi-monitor")
        self.interval_s = max(interval_ms, 50) / 1000.0
        self.duration_s = duration_s
        self.on_sample = on_sample
        # NB: имя атрибута выбрано `_stop_event` (НЕ `_stop`!) — вариант
        # `_stop` shadow'ит private `Thread._stop()` в Python 3.10+, И Python
        # вызывает этот метод во время Thread.join() через _wait_for_tstate_lock
        # (threading.py:~1118) → TypeError: 'Event' object is not callable.
        # Имя `_stop_event` не конфликтует с private Thread API.
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()
    
    def run(self) -> None:
        start = time.monotonic()
        deadline = start + self.duration_s if self.duration_s > 0 else None
        while not self._stop_event.is_set():
            sample = _nvidia_smi_query()
            if sample is not None:
                self.on_sample(sample, time.monotonic() - start)
            if deadline is not None and time.monotonic() >= deadline:
                break
            self._stop_event.wait(self.interval_s)


# ─── Strategy projection ─────────────────────────────────────────────────
def _compute_other(
    components: dict[str, float], which: str,
) -> float:
    """Coalesce per-card «other» footprint.

    Использует canonical key ``sage_attention_scratch`` (синхронизировано с
    ``core/memory_tracker.COMPONENT_FOOTPRINT_GB``). Если компоненты переданы
    в старой схеме (с ключом ``sage_scratch``) —backward-compatibility fallback
    через второй ``.get(...)``.
    """
    sage_scratch = components.get(
        "sage_attention_scratch",
        components.get("sage_scratch", 0.0),
    )
    if which == "cuda0":
        return (
            components.get("text_projection", 0.0)
            + components.get("video_vae", 0.0)
            + components.get("latent_upscaler", 0.0)
            + sage_scratch
        )
    return (
        components.get("audio_vae", 0.0)
        + sage_scratch
    )


def project_strategy(
    strategy: str,
    dit_gb: float,
    gemma_gb: float | None,
    cap0_gb: float,
    cap1_gb: float,
    components: dict[str, float],
) -> dict[str, Any]:
    """Return per-card projected GB + OK marks. Зеркалит
    ``core/memory_tracker.estimate_vram_budget._project``."""
    other0 = _compute_other(components, "cuda0")
    other1 = _compute_other(components, "cuda1")
    loras = components.get("loras_estimate", 0.0)
    gemma = gemma_gb or 0.0
    
    if strategy == "single_cuda0":
        c0, c1 = dit_gb + gemma + other0 + loras, other1
    elif strategy == "single_cuda1":
        c0, c1 = other0, dit_gb + gemma + other1 + loras
    elif strategy == "pipeline":  # DiT на cuda:1, Gemma на cuda:0
        # NB: loras УЧИТЫВАЮТСЯ на cuda:1 (туда, где живёт DiT) — это поведение
        # согласовано с core/memory_tracker._project("pipeline") (FIX MED-4 в
        # v0.2.1). До этого фикса standalone-CLI output отличался от in-package
        # pre-flight: standalone показывал pipeline как "OK" на edge-budget
        # конфигурациях с LoRAми, а реальный запуск вылетал в OOM.
        c0, c1 = gemma + other0, dit_gb + other1 + loras
    elif strategy == "blocks_50_50":
        c0, c1 = dit_gb / 2 + other0 + loras, dit_gb / 2 + gemma + other1
    elif strategy == "blocks_30_70":
        c0, c1 = (
            dit_gb * 0.3 + other0 + loras,
            dit_gb * 0.7 + gemma + other1,
        )
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")
    
    return {
        "cuda0_load_gb": round(c0, 3),
        "cuda1_load_gb": round(c1, 3),
        "cuda0_cap_gb":  round(cap0_gb, 3),
        "cuda1_cap_gb":  round(cap1_gb, 3),
        "cuda0_ok":      c0 <= cap0_gb,
        "cuda1_ok":      c1 <= cap1_gb,
        "ok":            (c0 <= cap0_gb) and (c1 <= cap1_gb),
    }


def recommend_strategy(
    dit_gb: float, gemma_gb: float | None,
    cap0_gb: float, cap1_gb: float,
    components: dict[str, float],
    preferred: str = "blocks_50_50",
) -> str:
    """First preference — ``preferred``; затем blocks_30_70 → pipeline → single_cuda0.
    Returns "FAILED" если ничего не помещается."""
    for strat in (preferred, "blocks_50_50", "blocks_30_70",
                  "pipeline", "single_cuda0"):
        if project_strategy(strat, dit_gb, gemma_gb,
                            cap0_gb, cap1_gb, components)["ok"]:
            return strat
    return "FAILED"


# ─── Output renderers ────────────────────────────────────────────────────
def _pct(num: float, cap: float) -> str:
    if cap <= 0:
        return "  ?  "
    return f"{(num / cap * 100.0):5.1f}%"


def _fmt_mib_to_gb(s: str) -> float:
    try:
        return int(s) / 1024.0
    except (ValueError, TypeError):
        return 0.0


def render_text(
    args: argparse.Namespace,
    gguf: GGUFScan | None,
    sts: SafetensorsScan | None,
    gpu_rows: list[dict[str, str]] | None,
    smi_samples: list[tuple[float, list[dict[str, str]]]],
) -> str:
    """Pretty output mirrors PLAN §5.1."""
    out: list[str] = []
    out.append("LTX-2 MultiGPU Diagnostic — pre-flight VRAM check")
    out.append("=" * 64)
    
    out.append("\n[Files]")
    if gguf is not None:
        fp4_extra = (
            f"  fp4={gguf.fp4_estimate_gb:.2f} GB"
            if gguf.fp4_estimate_gb is not None else ""
        )
        out.append(
            f"  DiT    {Path(gguf.path).name}  "
            f"size={gguf.file_gb:6.2f} GB  "
            f"tensors={gguf.n_tensors}  "
            f"quant={gguf.quant_name}"
        )
        out.append(
            f"         VRAM≈{gguf.quant_vram_estimate_gb:6.2f} GB (квант)  "
            f"fp16≈{gguf.fp16_estimate_gb:6.2f} GB (справочно){fp4_extra}"
        )
    else:
        out.append("  DiT    (not provided / not found)")
    if sts is not None:
        out.append(
            f"  Gemma  {Path(sts.path).name}  "
            f"size={sts.file_gb:6.2f} GB  "
            f"tensors={sts.n_tensors}  "
            f"fp16_proj={sts.fp16_estimate_gb:6.2f} GB"
        )
    else:
        out.append("  Gemma  (not provided / skipped)")
    
    out.append("\n[Hardware]")
    cap0_gb = cap1_gb = KAGGLE_VRAM_PER_T4_GB - KAGGLE_VRAM_RESERVED_GB  # Kaggle default
    if gpu_rows:
        for r in gpu_rows:
            cap = _fmt_mib_to_gb(r["memory_total_mib"])
            free = _fmt_mib_to_gb(r["memory_free_mib"])
            used = _fmt_mib_to_gb(r["memory_used_mib"])
            out.append(
                f"  cuda:{r['index']}  {r['name']}  "
                f"total={cap:5.2f} GB  "
                f"free={free:5.2f} GB  "
                f"used={used:5.2f} GB  "
                f"util={r['gpu_util_pct']}%"
            )
        cap0_gb = _fmt_mib_to_gb(gpu_rows[0]["memory_total_mib"])
        cap1_gb = (
            _fmt_mib_to_gb(gpu_rows[1]["memory_total_mib"])
            if len(gpu_rows) >= 2 else cap0_gb
        )
    else:
        out.append(
            "  (nvidia-smi unavailable; using Kaggle default "
            f"cap={KAGGLE_VRAM_PER_T4_GB - KAGGLE_VRAM_RESERVED_GB:.1f} ГБ / card — typical T4)"
        )
    
    out.append("\n[Other components — fixed estimates, GB]")
    for k, v in COMPONENT_FOOTPRINT_GB.items():
        out.append(f"  {k:>22s}: {v:5.2f}")

    # ── RAM Budget (Kaggle Edition) ─────────────────────────────────────
    if gguf is not None:
        ram_used = gguf.file_gb + PYTHON_COMFY_RAM_OVERHEAD_GB
        out.append(f"\n[RAM Budget]")
        out.append(
            f"  DiT mmap: {gguf.file_gb:.2f} GB + "
            f"overhead: {PYTHON_COMFY_RAM_OVERHEAD_GB:.2f} GB "
            f"= {ram_used:.2f} GB / {KAGGLE_SYSTEM_RAM_GB:.1f} GB RAM"
        )
        if ram_used > KAGGLE_SYSTEM_RAM_GB:
            out.append(
                f"  ⚠️  OOM RISK: {ram_used:.2f} GB > {KAGGLE_SYSTEM_RAM_GB:.1f} GB RAM! "
                f"mmap будет thrash'ить."
            )
        else:
            out.append(
                f"  ✓  RAM OK ({KAGGLE_SYSTEM_RAM_GB - ram_used:.1f} GB свободно)"
            )

    out.append("\n[Strategy projection]")
    # Kaggle Edition: используем QUANT-AWARE оценку VRAM, а не fp16.
    dit_gb = gguf.quant_vram_estimate_gb if gguf else 0.0
    gemma_gb = sts.fp16_estimate_gb if sts else None
    for strat in STRATEGIES:
        proj = project_strategy(
            strat, dit_gb, gemma_gb, cap0_gb, cap1_gb, COMPONENT_FOOTPRINT_GB,
        )
        ok0 = "OK " if proj["cuda0_ok"] else "OVR"
        ok1 = "OK " if proj["cuda1_ok"] else "OVR"
        out.append(
            f"  {strat:>14s}: cuda0={proj['cuda0_load_gb']:5.2f} GB "
            f"[{ok0} {_pct(proj['cuda0_load_gb'], cap0_gb)}/cap "
            f"{cap0_gb:.1f}]  |  "
            f"cuda1={proj['cuda1_load_gb']:5.2f} GB "
            f"[{ok1} {_pct(proj['cuda1_load_gb'], cap1_gb)}/cap "
            f"{cap1_gb:.1f}]"
        )
    
    rec = recommend_strategy(
        dit_gb, gemma_gb, cap0_gb, cap1_gb, COMPONENT_FOOTPRINT_GB,
        preferred=args.strategy,
    )
    out.append(f"\n[Recommendation] {rec}  "
               f"(преф: {args.strategy})")
    
    if gguf is not None and gguf.n_tensors == 0:
        out.append("\n⚠ GGUF header не распарсен — установлен ли `pip install gguf`?")
    if sts is not None and sts.n_tensors == 0:
        out.append("\n⚠ Safetensors header не распарсен — установлен ли "
                   "`pip install safetensors`?")

    if gguf is not None and gguf.quant_vram_estimate_gb > 12.0:
        out.append(
            f"\n💡 Квант {gguf.quant_name} — {gguf.quant_vram_estimate_gb:.1f} GB VRAM."
        )
        if gguf.quant_vram_estimate_gb > 14.0:
            out.append(
                "    На Kaggle T4×2 (14.5 GB каждая) — tight. "
                "Рассмотрите Q4_K_M (~12 GB VRAM) или Q3_K_XL (~10 GB VRAM)."
            )

    if smi_samples:
        out.append(
            f"\n[nvidia-smi samples] {len(smi_samples)} polls @ "
            f"{args.smi_poll_ms} ms each"
        )
        if args.verbose and smi_samples:
            t0 = smi_samples[0][0]
            for i, (t, rows) in enumerate(smi_samples[:10]):
                row_str = "  ".join(
                    f"cuda{r['index']}={_fmt_mib_to_gb(r['memory_used_mib']):.1f}"
                    f"/{_fmt_mib_to_gb(r['memory_total_mib']):.1f}GB"
                    f"({r['gpu_util_pct']}%)"
                    for r in rows
                )
                out.append(f"  t+{t - t0:6.2f}s : {row_str}")
    
    return "\n".join(out) + "\n"


# ─── CLI ─────────────────────────────────────────────────────────────────
def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="diagnostic.py",
        description="Pre-flight VRAM diagnostic for LTX 2.3 Hybrid Multi-GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/diagnostic.py\n"
            "  python scripts/diagnostic.py --unet lt2-3-q6_K.gguf --gemma gemma-3-12b-fp4.safetensors\n"
            "  python scripts/diagnostic.py --strategy pipeline\n"
            "  python scripts/diagnostic.py --smi-poll-ms 500 --smi-duration-s 60\n"
            "  python scripts/diagnostic.py --list\n"
        ),
    )
    p.add_argument("--unet", default="",
                   help="GGUF filename OR path (default: autodetect first .gguf)")
    p.add_argument("--gemma", default="",
                   help="Safetensors filename OR path (default: autodetect first in text_encoders/)")
    p.add_argument("--project-root", default="",
                   help="ComfyUI root (default: autodetect via CWD)")
    p.add_argument("--strategy", default="blocks_50_50", choices=STRATEGIES,
                   help="Preferred strategy (default: blocks_50_50)")
    p.add_argument("--smi-poll-ms", type=int, default=0,
                   help="nvidia-smi poll interval in ms "
                        "(0 = disabled, single probe; >0 — background poller)")
    p.add_argument("--smi-duration-s", type=int, default=0,
                   help="Stop polling after N seconds "
                        "(0 = single probe non-blocking)")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON instead of pretty text")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose per-component / per-sample breakdown")
    p.add_argument("--list", action="store_true",
                   help="List candidate GGUF / safetensors files under project-root")
    return p


def _list_files(root: Path | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"gguf": [], "safetensors": []}
    if root is None:
        return out
    out["gguf"] = sorted(
        str(p.relative_to(root)) for p in
        (root / "models" / "diffusion_models").glob("*.gguf")
    )
    for sub in ("text_encoders", "clip"):
        out["safetensors"] += sorted(
            str(p.relative_to(root)) for p in
            (root / "models" / sub).glob("*.safetensors")
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    
    project_root = _autodetect_project_root(args.project_root)
    
    # ── --list exit path ────────────────────────────────────────────────────
    if args.list:
        listing = _list_files(project_root)
        if args.json:
            print(json.dumps({"project_root": str(project_root), **listing},
                             indent=2))
        else:
            print(f"project_root: {project_root}")
            print(f"  GGUF ({len(listing['gguf'])}):")
            for p in listing["gguf"]:
                print(f"    {p}")
            print(f"  Safetensors ({len(listing['safetensors'])}):")
            for p in listing["safetensors"]:
                print(f"    {p}")
        return 0
    
    # ── Resolve file paths + scan headers ──────────────────────────────────
    gguf_path = _resolve_gguf(args.unet, project_root)
    sts_path = _resolve_safetensors(args.gemma, project_root)
    if args.unet and gguf_path is None:
        sys.stderr.write(
            f"ERROR: --unet ГДЕ {args.unet!r} не нашли ни в CWD, "
            f"ни в {project_root}/models/diffusion_models/\n"
            if project_root else
            f"ERROR: --unet {args.unet!r} не существует.\n"
        )
        return 3
    if args.gemma and sts_path is None:
        sys.stderr.write(f"WARN: --gemma {args.gemma!r} не найден — пропускаем.\n")
    
    try:
        gguf_scan = GGUFScan.scan(gguf_path) if gguf_path else None
    except Exception as exc:
        sys.stderr.write(f"ERROR: GGUF scan failed for {gguf_path}: {exc}\n")
        gguf_scan = None
    try:
        sts_scan = SafetensorsScan.scan(sts_path) if sts_path else None
    except Exception as exc:
        sys.stderr.write(f"WARN: safetensors scan failed for {sts_path}: {exc}\n")
        sts_scan = None
    
    # ── Hardware probe ─────────────────────────────────────────────────────
    gpu_rows = _nvidia_smi_query()
    cap0_gb = cap1_gb = KAGGLE_VRAM_PER_T4_GB - KAGGLE_VRAM_RESERVED_GB
    if gpu_rows:
        cap0_gb = _fmt_mib_to_gb(gpu_rows[0]["memory_total_mib"])
        cap1_gb = (
            _fmt_mib_to_gb(gpu_rows[1]["memory_total_mib"])
            if len(gpu_rows) >= 2 else cap0_gb
        )
    else:
        sys.stderr.write(
            "WARN: nvidia-smi недоступен (нет в PATH или нет CUDA) — "
            f"используются Kaggle дефолты cap={KAGGLE_VRAM_PER_T4_GB - KAGGLE_VRAM_RESERVED_GB:.1f} ГБ на карту.\n"
        )
    
    # ── Sми poll (single или background) ───────────────────────────────────
    smi_samples: list[tuple[float, list[dict[str, str]]]] = []
    
    if args.smi_poll_ms > 0 and args.smi_duration_s > 0:
        # Background polling на duration_s
        monitor = _SmiMonitor(
            args.smi_poll_ms, args.smi_duration_s,
            lambda row, t: smi_samples.append((t, row)),
        )
        monitor.start()
        try:
            # main thread ждёт; KeyboardInterrupt → cancel
            monitor.join(timeout=args.smi_duration_s + 5)
        except KeyboardInterrupt:
            monitor.stop()
            monitor.join(timeout=2.0)
    elif args.smi_poll_ms > 0:
        # Single probe, non-blocking
        single = _nvidia_smi_query()
        if single is not None:
            smi_samples.append((0.0, single))
    
    # ── Output ─────────────────────────────────────────────────────────────
    if args.json:
        dit_gb = gguf_scan.quant_vram_estimate_gb if gguf_scan else 0.0
        payload = {
            "project_root": str(project_root) if project_root else None,
            "files": {
                "unet":  asdict(gguf_scan) if gguf_scan else None,
                "gemma": asdict(sts_scan) if sts_scan else None,
            },
            "hardware": gpu_rows,
            "components_gb": COMPONENT_FOOTPRINT_GB,
            "kaggle_constants": {
                "system_ram_gb": KAGGLE_SYSTEM_RAM_GB,
                "vram_per_t4_gb": KAGGLE_VRAM_PER_T4_GB,
                "vram_reserved_gb": KAGGLE_VRAM_RESERVED_GB,
                "ram_overhead_gb": PYTHON_COMFY_RAM_OVERHEAD_GB,
            },
            "strategies": {
                s: project_strategy(
                    s,
                    dit_gb,
                    sts_scan.fp16_estimate_gb if sts_scan else None,
                    cap0_gb, cap1_gb, COMPONENT_FOOTPRINT_GB,
                )
                for s in STRATEGIES
            },
            "recommendation": recommend_strategy(
                dit_gb,
                sts_scan.fp16_estimate_gb if sts_scan else None,
                cap0_gb, cap1_gb, COMPONENT_FOOTPRINT_GB,
                preferred=args.strategy,
            ),
            "smi_samples": [
                {"t_s": round(t, 3), "rows": rows}
                for t, rows in smi_samples
            ],
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(render_text(args, gguf_scan, sts_scan, gpu_rows, smi_samples))
    
    # ── Exit code ──────────────────────────────────────────────────────────
    # Kaggle Edition: quant-aware оценка для exit-code логики.
    dit_gb = gguf_scan.quant_vram_estimate_gb if gguf_scan else 0.0
    gemma_gb = sts_scan.fp16_estimate_gb if sts_scan else None
    recommended = recommend_strategy(
        dit_gb, gemma_gb, cap0_gb, cap1_gb, COMPONENT_FOOTPRINT_GB,
        preferred=args.strategy,
    )
    if recommended == "FAILED":
        return 1
    if gpu_rows is None and gguf_scan is not None:
        return 2  # projection only — suggest retry on hardware
    return 0


if __name__ == "__main__":
    sys.exit(main())
