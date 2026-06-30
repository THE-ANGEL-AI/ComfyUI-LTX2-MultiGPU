#!/usr/bin/env python3
"""Download all required models for ComfyUI-LTX2-MultiGPU.

Скачивает 3 файла в правильные папки ComfyUI/models/:
  diffusion_models/ ← ltx-2.3-22b-distilled-UD-Q4_K_M.gguf  (~14 GB)
  text_encoders/    ← gemma_3_12B_it_fp4_mixed.safetensors     (~7.5 GB)
  vae/              ← diffusion_pytorch_model.safetensors      (~100 MB)

Запуск:
  python scripts/download_models.py                # авто-определение ComfyUI root
  python scripts/download_models.py --comfy-root D:\\ComfyUI  # явный путь
  python scripts/download_models.py --dry-run      # показать что будет скачано

Требования: huggingface_hub (pip install huggingface_hub)
Без него — скрипт выводит ручные команды для huggingface-cli.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# Конфигурация моделей
# ═══════════════════════════════════════════════════════════════════════════

MODELS: list[dict] = [
    {
        "name": "LTX-Video 2.3 22B distilled GGUF (UD-Q4_K_M)",
        "repo_id": "unsloth/LTX-2.3-GGUF",
        "filename": "distilled/ltx-2.3-22b-distilled-UD-Q4_K_M.gguf",
        "folder": "diffusion_models",
        "local_name": "ltx-2.3-22b-distilled-UD-Q4_K_M.gguf",
        "size_gb": 14.0,
        "emoji": "🔀",
        "help": "DiT модель для HybridSplitLoader. Q4_K_M (~4.2 bpw) влезает в 14.5 GB T4.",
    },
    {
        "name": "Gemma 3 12B Instruct FP4 (text encoder)",
        "repo_id": "Comfy-Org/ltx-2",
        "filename": "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        "folder": "text_encoders",
        "local_name": "gemma_3_12B_it_fp4_mixed.safetensors",
        "size_gb": 7.5,
        "emoji": "📝",
        "help": (
            "Текстовый энкодер для Dual CLIP Загрузчика. "
            "Этот же файл используется как clip_name1 и projection_name "
            "(comfy.sd.load_clip загружает оба из одного safetensors)."
        ),
    },
    {
        "name": "LTX-Video VAE",
        "repo_id": "Lightricks/LTX-Video",
        "filename": "vae/diffusion_pytorch_model.safetensors",
        "folder": "vae",
        "local_name": "ltx_video_2_3_vae.safetensors",
        "size_gb": 0.1,
        "emoji": "🖼️",
        "help": "VAE для decode/encode. ~100 MB, загружается VAE Загрузчиком (GPU).",
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# ComfyUI root detection
# ═══════════════════════════════════════════════════════════════════════════


def _find_comfy_root() -> Optional[Path]:
    """Ищет корень ComfyUI поднимаясь вверх от scripts/ до нахождения main.py.

    Не полагается на фиксированную глубину — если scripts/ переместят
    глубже или мельче, поиск всё равно найдёт ComfyUI root.
    """
    current = Path(__file__).resolve().parent
    for _ in range(8):  # максимум 8 уровней вверх
        if (current / "main.py").exists() or (current / "nodes.py").exists():
            # Доп. проверка: это похоже на ComfyUI (не на другой проект)
            if (current / "comfy").is_dir() or (current / "custom_nodes").is_dir():
                return current
        # Fallback: models/ рядом с main.py
        if (current / "models").is_dir() and (
            (current / "main.py").exists() or (current / "nodes.py").exists()
        ):
            return current
        current = current.parent
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Download helpers
# ═══════════════════════════════════════════════════════════════════════════


def _has_huggingface_hub() -> bool:
    """Проверяет доступен ли huggingface_hub."""
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


def _download_with_hub(
    repo_id: str,
    filename: str,
    local_dir: Path,
    local_name: str,
) -> bool:
    """Скачивает файл через huggingface_hub.hf_hub_download.

    Returns True если успешно, False если ошибка.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume=True,
        )
        # Переименовываем в локальное имя если отличается
        target = local_dir / local_name
        if Path(downloaded_path) != target:
            import shutil
            shutil.move(str(downloaded_path), str(target))
        # Убираем пустую родительскую папку после move (например distilled/)
        try:
            leftover = local_dir / Path(filename).parent
            if leftover != local_dir and leftover.exists():
                import shutil as _shutil
                _shutil.rmtree(str(leftover), ignore_errors=True)
        except Exception:
            pass
        return True
    except HfHubHTTPError as exc:
        print(f"  ERROR: {exc}")
        return False
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False


def _print_manual_command(
    repo_id: str,
    filename: str,
    local_dir: Path,
    local_name: str,
) -> None:
    """Выводит ручную команду для huggingface-cli."""
    _safe_print(f"  huggingface-cli download {repo_id} {filename} --local-dir \"{local_dir}\"")
    if local_name != Path(filename).name:
        _safe_print(f"  # затем переименуйте в {local_name}")


def _format_gb(size_gb: float) -> str:
    """Форматирует размер в человекочитаемый вид."""
    if size_gb < 1:
        return f"{size_gb * 1000:.0f} MB"
    return f"{size_gb:.1f} GB"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def _safe_print(*args, **kwargs) -> None:
    """print с защитой от UnicodeEncodeError на Windows cp1251 терминалах."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Фолбек: заменяем непечатные символы на '?'
        safe_args = [
            str(a).encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
                sys.stdout.encoding or "utf-8", errors="replace"
            )
            if isinstance(a, str) else a
            for a in args
        ]
        try:
            print(*safe_args, **kwargs)
        except Exception:
            # Совсем глухой fallback
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            print(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Скачать модели для ComfyUI-LTX2-MultiGPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python scripts/download_models.py\n"
            "  python scripts/download_models.py --comfy-root D:\\ComfyUI\n"
            "  python scripts/download_models.py --dry-run\n"
            "  python scripts/download_models.py --only diffusion_models\n"
        ),
    )
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=None,
        help="Путь к корню ComfyUI (авто-определение если не указан)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать что будет скачано без загрузки",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=["diffusion_models", "text_encoders", "vae"],
        help="Скачать только одну категорию моделей",
    )
    args = parser.parse_args()

    # ── Определяем ComfyUI root ──────────────────────────────────────────
    comfy_root = args.comfy_root or _find_comfy_root()
    if comfy_root is None:
        _safe_print("[FAIL] Не удалось найти корень ComfyUI.")
        _safe_print("       Укажите путь явно: --comfy-root ПУТЬ")
        return 1

    models_dir = comfy_root / "models"
    if not models_dir.is_dir():
        _safe_print(f"[WARN] Папка models/ не найдена в {comfy_root}")
        _safe_print(f"       Создаю: {models_dir}")
        models_dir.mkdir(parents=True, exist_ok=True)

    if not comfy_root.exists():
        _safe_print(f"[FAIL] Указанный ComfyUI root не существует: {comfy_root}")
        return 1

    _safe_print(f"ComfyUI root: {comfy_root}")
    _safe_print(f"Models dir:   {models_dir}")
    _safe_print()

    # ── Проверяем инструмент загрузки ────────────────────────────────────
    has_hub = _has_huggingface_hub()
    if not has_hub:
        _safe_print("[WARN] huggingface_hub не установлен.")
        _safe_print("       Установите: pip install huggingface_hub")
        _safe_print("       Или используйте huggingface-cli вручную (команды ниже).")
        _safe_print()

    # ── Фильтруем модели ─────────────────────────────────────────────────
    to_download = MODELS
    if args.only:
        to_download = [m for m in MODELS if m["folder"] == args.only]
        if not to_download:
            _safe_print(f"[FAIL] Нет моделей для категории '{args.only}'")
            return 1

    total_gb = sum(m["size_gb"] for m in to_download)
    _safe_print(f"{'[DRY RUN] ' if args.dry_run else ''}Моделей к загрузке: {len(to_download)}")
    _safe_print(f"   Общий размер: ~{_format_gb(total_gb)}")
    _safe_print()

    # ── Загружаем / выводим ──────────────────────────────────────────────
    success = 0
    failed = 0

    for i, model in enumerate(to_download, 1):
        folder = models_dir / model["folder"]
        local_path = folder / model["local_name"]

        emoji = model["emoji"]
        name = model["name"]
        size_str = _format_gb(model["size_gb"])

        _safe_print(f"[{i}/{len(to_download)}] {emoji} {name}")
        _safe_print(f"           Размер: {size_str}")
        _safe_print(f"           Папка:  {model['folder']}/")
        _safe_print(f"           Файл:   {model['local_name']}")

        if args.dry_run:
            if local_path.exists():
                _safe_print(f"           [OK] Уже скачан ({_format_gb(local_path.stat().st_size / 1e9)} на диске)")
            else:
                _safe_print(f"           [-->] Будет скачан из {model['repo_id']}")
            _safe_print()
            continue

        # Проверяем что файл уже есть
        if local_path.exists():
            existing_gb = local_path.stat().st_size / 1e9
            if existing_gb > model["size_gb"] * 0.9:  # >= 90% ожидаемого размера
                _safe_print(f"           [OK] Уже скачан ({_format_gb(existing_gb)} на диске) — пропускаю")
                _safe_print()
                success += 1
                continue
            else:
                _safe_print(f"           [WARN] Файл есть но мал ({_format_gb(existing_gb)} vs ~{size_str}) — перекачиваю")
                local_path.unlink()

        # Создаём папку
        folder.mkdir(parents=True, exist_ok=True)

        if has_hub:
            _safe_print(f"           [-->] Скачиваю из {model['repo_id']}...")
            if _download_with_hub(
                repo_id=model["repo_id"],
                filename=model["filename"],
                local_dir=folder,
                local_name=model["local_name"],
            ):
                _safe_print(f"           [OK] Готово!")
                success += 1
            else:
                _safe_print(f"           [FAIL] Не удалось скачать")
                failed += 1
        else:
            _safe_print(f"           Ручная команда:")
            _print_manual_command(
                repo_id=model["repo_id"],
                filename=model["filename"],
                local_dir=folder,
                local_name=model["local_name"],
            )
            # Проверяем что файл теперь есть (могли скачать вручную)
            if local_path.exists():
                _safe_print(f"           [OK] Файл уже на месте")
                success += 1
            else:
                failed += 1
        _safe_print()

    # ── Итог ─────────────────────────────────────────────────────────────
    _safe_print("=" * 60)
    _safe_print(f"[OK] Успешно: {success}")
    if failed:
        _safe_print(f"[FAIL] Не удалось: {failed}")
    if not has_hub and not args.dry_run:
        _safe_print()
        _safe_print("Установите huggingface_hub для автоматической загрузки:")
        _safe_print("   pip install huggingface_hub")
    _safe_print()
    _safe_print("Структура после загрузки:")
    for model in to_download:
        _safe_print(f"   {models_dir / model['folder'] / model['local_name']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
