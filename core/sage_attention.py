"""core/sage_attention.py — Интеграция SageAttention-SM75 для T4 (Turing SM75).

White Paper §6: SageAttention заменяет ``scaled_dot_product_attention`` на
квантованное внимание (INT8 QK^T, FP16 PV), давая ~1.5x ускорение end-to-end
на T4. Интеграция через ``patcher.model_options['attention_patch']`` —
чистый ComfyUI-путь без глобального monkey-patch.

``get_sageattn_patch(verbose)``:
  - Пробует ``import sageattn`` (лениво, не на верхнем уровне)
  - Если успешно → возвращает ``{\"default\": wrapper}`` для attention_patch
  - Если нет → возвращает ``{}`` + печатает WARN
  - Идемпотентен, безопасен для вызова без GPU
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_sageattn_patch", "is_sageattn_available"]


def is_sageattn_available() -> bool:
    """Проверяет, можно ли импортировать sageattn (без побочных эффектов)."""
    try:
        import sageattn  # noqa: F401
        return True
    except ImportError:
        return False


def get_sageattn_patch(verbose: bool = False) -> dict[str, Any]:
    """Возвращает attention_patch dict для ComfyUI model_options.

    Если sageattn (SageAttention-SM75) установлен — возвращает
    ``{\"default\": wrapper}``, где wrapper оборачивает вызов
    ``sageattn.sageattn(q, k, v)``.

    Если sageattn НЕ установлен — возвращает ``{}`` и печатает
    WARN (unconditional, как и другие WARN в проекте).

    Совместимость:
      - Работает на T4 (SM75) через Triton fallback path
      - На других GPU (Ampere/Ada/Hopper) пробрасывает вызов в
        нативный sageattn (если установлен)
      - Без GPU (тесты) — тихо возвращает {}
    """
    try:
        import sageattn  # noqa: F811 — ленивый импорт внутри функции
    except ImportError:
        print(
            "[ComfyUI-LTX2-MultiGPU] WARN: sageattn (SageAttention-SM75) "
            "не установлен. Установите: pip install sageattn "
            "или клонируйте https://github.com/THE-ANGEL-AI/SageAttention-SM75\n"
            "  Стандартное внимание будет использовано (без ускорения)."
        )
        return {}

    # Проверяем, что sageattn имеет нужный API.
    _attn_fn = getattr(sageattn, "sageattn", None)
    if _attn_fn is None:
        print(
            "[ComfyUI-LTX2-MultiGPU] WARN: sageattn установлен, "
            "но не имеет sageattn.sageattn() — возможно, несовместимая версия. "
            "Патч внимания НЕ применён."
        )
        return {}

    def _sage_attn_wrapper(
        q: Any, k: Any, v: Any, extra_options: Any = None
    ) -> Any:
        """Обёртка для ComfyUI attention_patch.

        Принимает q, k, v (тензоры) и extra_options (dict от ComfyUI),
        возвращает результат sageattn(q, k, v).
        extra_options принимается для совместимости с сигнатурой
        ComfyUI, но не используется (SageAttention сам управляет
        параметрами внимания через тензорные свойства).
        """
        return _attn_fn(q, k, v)

    if verbose:
        try:
            ver = getattr(sageattn, "__version__", "?")
        except Exception:  # noqa: BLE001
            ver = "?"
        print(
            f"[ComfyUI-LTX2-MultiGPU] SageAttention-SM75 v{ver} "
            f"активирован через attention_patch."
        )

    return {"default": _sage_attn_wrapper}
