"""core/gguf_reader.py — обёртка над gguf.GGUFReader (header-only режим + dequant helpers).

Использует стандартный `gguf` Python пакет (https://pypi.org/project/gguf/), который
читает .gguf файлы через mmap без полной загрузки весов в RAM. Это даёт:

  - `list_gguf_tensors(path)` — читает только header (tensor names + dtypes + shape)
  - `dequant_to_fp16(path)` — загружает все тензоры как fp16 torch.Tensor

⚠️ Реальные имена тензоров после городского city96-ComfyUI-GGUF loader:
  см. MODEL_FACTS §3 — `model.diffusion_model.transformer_blocks.{0..43}.*`
  (НЕ `.layers.*` — именно это DisTorch2 и не находит).
"""

from __future__ import annotations

from typing import Any

try:
    import gguf  # type: ignore[import-not-found]  # pip install gguf (см. requirements.txt)
except Exception:  # noqa: BLE001
    gguf = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]


__all__ = ["list_gguf_tensors", "dequant_to_fp16", "read_gguf_header"]


def _reader(path: str):  # pragma: no cover - thin wrapper
    """Открывает GGUF-файл в режиме mmap (header-only доступен)."""
    if gguf is None:
        raise RuntimeError(
            "Пакет `gguf` не установлен. Запустите: "
            "pip install -r custom_nodes/ComfyUI-LTX2-MultiGPU/requirements.txt"
        )
    # gguf.GGUFReader(path) открывает файл с mmap=True по умолчанию.
    # Пока не обращаемся к tensor.data, файл НЕ грузится в RAM.
    return gguf.GGUFReader(path)


def read_gguf_header(path: str) -> dict[str, Any]:
    """Читает ТОЛЬКО header (без загрузки весов) — для memory_tracker.py.

    Возвращает dict с полями:
      - tensor_count: int
      - tensors: список {name, n_elements, tensor_type (e.g. Q5_K), shape}
      - bpe: bytes per element (расчётная оценка VRAM)
    """
    reader = _reader(path)
    out_tensors: list[dict[str, Any]] = []
    total_elements = 0
    for t in reader.tensors:
        out_tensors.append(
            {
                "name": t.name,
                "n_elements": int(t.n_elements),
                "tensor_type": str(t.tensor_type.name) if t.tensor_type is not None else "?",
                "shape": list(t.shape) if t.shape is not None else [],
            }
        )
        total_elements += int(t.n_elements)

    return {
        "tensor_count": len(out_tensors),
        "total_elements": total_elements,
        "tensors": out_tensors,
    }


def list_gguf_tensors(path: str) -> list[str]:
    """Возвращает список имён тензоров (header-only, без dequant)."""
    return [t["name"] for t in read_gguf_header(path)["tensors"]]


# Ориентировочный порог: при ~8 млрд элементов (~16 ГБ fp16) функция
# становится опасной для 15 ГБ GPU. Рекомендуется использовать только как
# diagnostic для tensor-names + shapes; для split используется city96 lazy loader.
_DIAG_MAX_ELEMENTS_HARD_LIMIT = 8_000_000_000  # ~16 GB fp16


def dequant_to_fp16(path: str) -> dict[str, Any]:
    """Читает GGUF и возвращает dict[имя] = fp16 torch.Tensor.

    ⚠️ NUCLEAR WARNING:
      22B DiT в Q5_K_M = 7.5 млрд элементов → ~15 ГБ fp16. На T4 15 ГБ это
      BORDER-LINE OOM с учётом activations. Для production используйте city96
      UnetLoaderGGUF (lazy GGMLOps) + split по блокам.
      Эта функция — только для диагностики / unit-test малых моделей.

    Raises:
      RuntimeError если total_elements > _DIAG_MAX_ELEMENTS_HARD_LIMIT.
    """
    if torch is None:
        raise RuntimeError("torch не установлен")
    reader = _reader(path)
    total_elt = sum(int(t.n_elements) for t in reader.tensors)
    if total_elt > _DIAG_MAX_ELEMENTS_HARD_LIMIT:
        raise RuntimeError(
            f"GGUF {path} имеет {total_elt:,} elements (~{total_elt*2/1024**3:.1f} ГБ fp16). "
            "Превышает _DIAG_MAX_ELEMENTS_HARD_LIMIT; используйте city96 "
            "UnetLoaderGGUF + split, а не dequant_to_fp16 в полном объёме. "
            "Если вам действительно нужна полная dequant, поднимите порог явно."
        )

    out: dict[str, Any] = {}
    for t in reader.tensors:
        try:
            # `gguf.dequantize` правильно раскодирует Q5_K_M / Q4_K_M и др.
            # При raw `astype('float16')` packed-bits будут reinterpreted как garbage.
            import numpy as np
            np_arr = gguf.dequantize(t.data, t.tensor_type)  # -> float32/float16 numpy
            np_arr = np_arr.astype("float16", copy=False)
            tensor = torch.from_numpy(np_arr)
            out[t.name] = tensor.to(torch.float16)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[ComfyUI-LTX2-MultiGPU] WARN: cannot dequant {t.name} "
                f"(type={t.tensor_type.name if t.tensor_type else '?'}): {exc}"
            )
            continue
    return out


__all__ = ["list_gguf_tensors", "dequant_to_fp16", "read_gguf_header"]
