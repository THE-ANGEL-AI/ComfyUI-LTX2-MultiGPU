"""ComfyUI-LTX2-MultiGPU — Hybrid Multi-GPU split for LTX 2.3 GGUF.

═══════════════════════════════════════════════════════════════════════════
  АВТОР / AUTHOR:  THE-ANGEL-AI  (The Angel Studio)
  Repo:            https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU
  Sponsor:         https://boosty.to/the_angel/donate
  License:         GPL-3.0-or-later (см. LICENSE)
  Display category: "THE-ANGEL-AI" (7 нод: 🧠 LTX2 GGUF / 📝 Dual CLIP / 💾 Диагностика / 🎯 Стратегия / 🅿️ Парковка / ⚡ SageAttention / 🎨 LTX2 VAE)

  Независимая разработка для конфигурации 2×15 ГБ (T4×2 / Kaggle) с
  долгой оптимизацией под этот железный профиль (см. CHANGELOG).
  Используются как референс: city96/ComfyUI-GGUF (GGUF loader), и
  pollockjj/ComfyUI-MultiGPU (MultiGPU patterns) — полные аттрибуции
  в README → Credits. Это НЕ форк каких-либо из тех проектов.
═══════════════════════════════════════════════════════════════════════════

Правило R6 (agent-rules): каждый импорт обёрнут в try/except — сбой в одном
узле НЕ должен ломать загрузку всего пакета (видно в меню Add Node).
"""

__version__ = "0.6.0-pre"
__author__ = "The Angel Studio"
__author_email__ = "gi.the.angel@gmail.com"
__author_github__ = "THE-ANGEL-AI"
__repo__ = "https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU"

# FIX (v0.6.1-pre): sys.path shim для совместимости с ComfyUI load- путями,
# которые НЕ добавляют base_dir пакета в sys.path автоматически
# (например, V3 imports / zip-installs / Kaggle runtime).
# Используем ``append`` (не ``insert(0, ...)``) — чтобы не ломать приоритеты
# upstream-пакетов (comfy/, torch/) в случае коллизии имени ``core``.
# Тесты в tests/ продолжат работать: они импортируют ``from core import ...``
# после этой строчки, base_dir уже известен интерпретатору. Primary
# motivation — Kaggle runtime + ComfyUI V3 loader, которые НЕ добавляют
# package dir в sys.path автоматически → ModuleNotFoundError: 'core'.
import sys as _sys
import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _BASE_DIR not in _sys.path:
    _sys.path.append(_BASE_DIR)

# Консольный баннер — opt-in через env var. По умолчанию тихо, чтобы НЕ
# засорять stdout при штатной загрузке пакета в ComfyUI. Установите
# ``LTX2_MULTIGPU_VERBOSE=1`` если хотите видеть attribution при старте.
import os as _os
if _os.environ.get("LTX2_MULTIGPU_VERBOSE") == "1":
    try:
        print(
            f"[ComfyUI-LTX2-MultiGPU v{__version__}] "
            f"author={__author__!r} github={__author_github__!r} "
            f"repo={__repo__}"
        )
    except Exception:  # noqa: BLE001
        # stdout может быть unavailable в редких env (frozen exe, systemd)
        pass

NODE_CONFIG: list[dict] = []
# Заполняется автоматически из nodes.py в _build_config() ниже.


def _build_config() -> None:
    """Собирает NODE_CONFIG из узлов, импортируемых безопасно (лениво).

    Каждая нода-класс должна иметь атрибуты:
      - NODE_ID       (str) — уникальный ключ для NODE_CLASS_MAPPINGS
      - DISPLAY_NAME  (str) — человекочитаемое имя в меню

    ⚠️ CRITICAL: Используем ОТНОСИТЕЛЬНЫЙ импорт ``.nodes``, а НЕ
    ``importlib.import_module("nodes")`` — последний в ComfyUI runtime
    находит ROOT ``nodes.py`` (уже в sys.modules) вместо нашего!

    NEW (v0.6.0-pre, reviewer-minimax-m3 item #2): clear _GEMMA_CACHE at
    package import. Drops any stale entries from prior-version 5-tuple
    keys that would orphan under current 4-tuple key contract
    (encoder_name, projection_name, donor_device, eject_models).
    """
    # NEW (v0.6.0-pre): _GEMMA_CACHE orphan cleanup at package load.
    # Wrapped in try/except so failure doesn't block package import (R6).
    try:
        from .core.gguf_split import clear_gemma_cache  # type: ignore[import-not-found]
        clear_gemma_cache()
    except Exception:  # noqa: BLE001
        # Best-effort cleanup; if core module is unavailable (test env),
        # silently skip — cache will work on next load.
        pass

    import inspect

    try:
        from . import nodes as _nodes
    except Exception as exc:  # noqa: BLE001
        print(f"[ComfyUI-LTX2-MultiGPU] Failed to import '.nodes': {exc}")
        import traceback as _tb
        _tb.print_exc()
        return

    # Собираем безопасные ссылки; битые классы не ломают пакет
    for _attr_name, attr_value in vars(_nodes).items():
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

# WEB_DIRECTORY объявляется только при наличии web/ — здесь его нет.

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
    "__author__",
]
