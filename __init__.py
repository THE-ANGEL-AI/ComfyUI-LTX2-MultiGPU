"""ComfyUI-LTX2-MultiGPU — Hybrid Multi-GPU split for LTX 2.3 GGUF.

Author: The Angel Studio <gi.the.angel@gmail.com>
License: MIT

Правило R6 (agent-rules): каждый импорт обёрнут в try/except — сбой в одном
узле НЕ должен ломать загрузку всего пакета (видно в меню Add Node).
"""

# VERSION синхронизирован с pyproject.toml (release v0.2.0, 2026-06-30)
__version__ = "0.2.0"
__author__ = "The Angel Studio"

NODE_CONFIG: list[dict] = [
    # Заполняется автоматически из nodes.py — см. _generate ниже.
]


def _build_config() -> None:
    """Собирает NODE_CONFIG из узлов, импортируемых безопасно (лениво).

    Каждая нода-класс должна иметь атрибуты:
      - NODE_ID       (str) — уникальный ключ для NODE_CLASS_MAPPINGS
      - DISPLAY_NAME  (str) — человекочитаемое имя в меню
    """
    import importlib
    import inspect

    for mod_name in ("nodes",):
        try:
            module = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001 — широкий except защищает пакет от сбоя
            print(f"[ComfyUI-LTX2-MultiGPU] Failed to import '{mod_name}': {exc}")
            # Полный traceback — иначе в консоли ComfyUI видна только
            # короткая строка exc и причина сбоя теряется (R6 хоть и
            # защищает пакет, но скрывать traceback — это анти-паттерн).
            import traceback as _tb
            _tb.print_exc()
            continue

        # Собираем безопасные ссылки; битые классы не ломают пакет
        for _attr_name, attr_value in vars(module).items():
            if not inspect.isclass(attr_value):
                continue
            cls_id = getattr(attr_value, "NODE_ID", None)
            if not isinstance(cls_id, str) or not cls_id.startswith("LTX2_MultiGPU_"):
                continue
            NODE_CONFIG.append(
                {
                    "id": cls_id,
                    "class": attr_value,
                    "name": getattr(attr_value, "DISPLAY_NAME", cls_id),
                }
            )


_build_config()

NODE_CLASS_MAPPINGS: dict[str, type] = {cfg["id"]: cfg["class"] for cfg in NODE_CONFIG}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {cfg["id"]: cfg["name"] for cfg in NODE_CONFIG}

# WEB_DIRECTORY обявляется только при наличии web/ — здесь его нет.

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
    "__author__",
]
