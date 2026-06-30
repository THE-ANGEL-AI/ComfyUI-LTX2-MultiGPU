# История изменений (Changelog)

Все значимые изменения в этом проекте документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
и этот проект придерживается [Semantic Versioning](https://semver.org/lang/ru/).

---

## [v0.2.0] — 2026-06-30

### License (LICENSING)

- **CHANGED**: переход с **MIT License** на **GNU GPL v3-or-later**. Solo copyright
  holder = The Angel Studio, переход унилатеральный (perpetual MIT grant позволяет
  автору пере-licence собственные копии на любые совместимые conditions). Полный
  текст — в `LICENSE` (SPDX-License-Identifier: `GPL-3.0-or-later`); classifier
  `License :: OSI Approved :: ... GPLv3+` добавлен в `pyproject.toml`.
- **CHANGED**: README раздел «Credits» + новый раздел «Лицензия и юридические
  моменты» — явно упоминают GPL-3.0-or-later для кода и отдельные licenses
  (LTX-Video Community License / Gemma License) для весов.

### Documentation (документация)

- **CHANGED**: README полностью переведён на **русский** с упрощённым языком для
  обычных пользователей (раздел «Что это и зачем — простыми словами») при
  сохранённой технической глубине:
  - VRAM-диаграмма ASCII с русскими labels («DiT блоки», «Gemma 3 12B FP4», «~9 ГБ», …).
  - Стратегии `blocks_50_50` / `blocks_30_70` / `pipeline` / `single_cuda0/1`.
  - GGUF-квантизации (Q4_K_M / Q5_K_M / Q6_K / Q3 / Q2) с approximate file sizes.
  - Hardware tested (T4×2 Kaggle / RTX 4090×2 / A5000+3090 asymmetric).
  - Советы и грабли (`eject_models`, prompt overflow на cuda:1, upscale OOM,
    live `nvidia-smi` через MemoryDiagnostics).

### Funding (финансирование)

- **ADDED**: `submodule/.github/FUNDING.yml` — нативная GitHub-кнопка Sponsor →
  Boosty (`custom: ["https://boosty.to/the_angel/donate"]`). GitHub читает этот
  файл из root репозитория и рендерит Sponsor-кнопку в правом sidebar.
- **ADDED**: README — Boosty-badge `[![Sponsor: Boosty](orange.svg)]` в шапке.
- **ADDED**: README — секция «Если хотите поддержать проект 💚» в начале + повторная
  CTA-секция «Поддержать проект (повтор)» перед Credits. Деньги идут на оплату
  GPU-часов Kaggle / Colab для тестов разных конфигураций.

### Fixed (стабилизация UI-стороны)

- **FIXED** (`c47a98a`): **Degenerate-WARN UNCONDITIONAL**. WARN о single-GPU setup
  (`secondary_dev == primary_dev`) и `effective_donor == primary_dev` теперь
  печатается **всегда**, не только при `verbose_log=True` в ноде Device Strategy
  Switch. Убирает silent normalisation, улучшает UX для пользователей на
  single-GPU машинах (видят явное WARN вместо тихого fallback).
- **FIXED** (`cab22dc`): **`apply_strategy` verbose-routing**. Нода Device Strategy
  Switch прокидывает свой `verbose_log` widget параметр в
  `core.gguf_split.apply_strategy(verbose=verbose_log)`. Forward verbose без
  re-gating уже-unconditional WARN (`# noqa: ARG003` для backward-compat signature).
- **FIXED** (`d9d0d6a`, **FIX LOW #4**): **`_cuda_donor_choices`** helper в
  module-level. Заменяет два class-level `_DONOR_DEVICE_CHOICES_*` tuples.
  Всегда возвращает `["auto", "cuda:0", "cuda:1"]` baseline (даже при `torch is
  None`); плюс `cuda:0..N` от `torch.cuda.device_count()` если torch+Cuda
  доступны. Tightened `except (RuntimeError, AssertionError)` вместо broad
  `except Exception`.
- **FIXED** (`d9d0d6a`, **FIX MEDIUM_apply_strategy**): **`DeviceStrategy.apply_strategy`
  verbose widget wiring**. Пробрасывает widget `verbose_log` в
  `core.apply_strategy(verbose=...)`. Закрывает loop между UI-виджетом и core-функцией.
- **FIXED** (`d9d0d6a`, **FIX R4 contract**): **`MemoryDiagnostics.diagnose` returns
  `(report,)`** 1-tuple. Совместимо с ComfyUI V1 unpacking `val, = node.FUNCTION(...)`.
  Устраняет `ValueError: too many values to unpack (expected 1)` на Kaggle zero-image
  и старых ComfyUI builds. Console preview остаётся через `print()` в ComfyUI log.

### Security (copyleft implications)

- **SECURITY (BREAKING for downstream)**: переход MIT → GPL-3.0-or-later создаёт
  **copyleft obligations** для downstream-ов. Любая копия проекта, полученная
  **после релиза v0.2.0 (2026-06-30)**, должна распространяться также под
  GPL-3.0-or-later (или совместимой лицензией). Старые форки, склонированные до
  2026-06-30, сохраняют MIT-лицензию для своего снимка (perpetual grant), но
  upstream pull после этой даты обязывает ребрендинг под GPL-3.0.
- Solo copyright holder = The Angel Studio; relicense унилатеральный.
- SPDX-License-Identifier в корне репо: `GPL-3.0-or-later`. Полный текст — в
  `LICENSE`; машиночитаемый classifier — в `pyproject.toml`.

### Compatibility note

- **ADDED**: явное замечание о компонентах в `LICENSE`: веса LTX 2.3 (Lightricks,
  LTX License) и Gemma 3 (Google, Gemma License) — собственных conditions, **не**
  GPL-3.0. Только код этого проекта под GPL-3.0-or-later. Перед коммерческим
  использованием генерации убедитесь в соблюдении license term чекпоинтов.

---

## [v0.1.0] — 2026-06-29

### First public release

- **ADDED**: `LTX2_MultiGPU_HybridSplitLoader` — загружает LTX 2.3 22B GGUF, делит
  44 DiT блока между двумя картами (`blocks_50_50` default), cross-card
  forward-hook для передачи скрытых состояний между картами между шагами KSampler.
- **ADDED**: `LTX2_MultiGPU_GemmaHybridLoader` — загружает Gemma 3 12B FP4 +
  `text_projection` как единый CLIP, кладёт на нужные карты.
- **ADDED**: `LTX2_MultiGPU_MemoryDiagnostics` — пред-полётный VRAM-чек +
  projection + `nvidia-smi` снапшот.
- **ADDED**: `LTX2_MultiGPU_DeviceStrategy` — hot-swap стратегии split'а
  (`blocks_30_70` / `pipeline` / `single_cuda0/1`) без перезагрузки модели.
- **ADDED**: `core/gguf_split.py` — GGUF-сплиттер + forward-hook установка для
  `apply_strategy`.
- **ADDED**: `core/gguf_reader.py` — низкоуровневый GGUF tensor reader (для
  лоадера).
- **ADDED**: `core/memory_tracker.py` — pre-flight VRAM projection
  (`estimate_vram_budget`).
- **ADDED**: README в **English** с полной архитектурой, VRM layout ASCII,
  install steps (Windows PowerShell / Linux bash / Colab / Kaggle), hardware
  tested table (T4×2 / RTX 4090×2 / A5000+3090), tips (eject_models / nvidia-smi /
  upscale OOM), Verified GGUF quants table, Compatibility note для LTX-Video
  Community License.
- **ADDED**: LICENSE (MIT). SPDX-License-Identifier: MIT.

---

## [pre-0.1.0] — разработка (phase 1 + phase 2 + phase 3)

### Phase 2 — core foundation (`d8e0a7d`)

- **ADDED**: `core/gguf_split.py` — GGUF-сплиттер + forward-hook установка
  (vector-aware dtype detection для `_move_param` / `_move_buffer`, kwargs hooks
  `with_kwargs=True`, lock `.to` dtype, degenerate guards `if torch is None`,
  `_install_cross_device_hook`).
- **ADDED**: `core/gguf_reader.py` — низкоуровневый GGUF tensor reader.
- **ADDED**: `core/memory_tracker.py` — pre-flight VRAM projection
  (`estimate_vram_budget`).
- **ADDED**: `__init__.py` — module entry с проверкой ComfyUI-folder paths и
  `FOLDER_PATHS_OK` флагом graceful degradation.

### Phase 3 — LTX-Video-specific nodes (`f8d553d`)

- **ADDED**: `nodes.py` с четырьмя нодами (`HybridSplitLoader` /
  `GemmaHybridLoader` / `MemoryDiagnostics` / `DeviceStrategy`).
- **ADDED**: README rewrite в OSS-style: hero, badges (License / ComfyUI / Python /
  Version), Quick-start, VRAM ASCII diagram, Hardware tested table, Tips, GGUF
  quants, Compatibility note.

### Архитектурный контекст (Why we exist)

- **Проблема**: `city96/ComfyUI-GGUF` деквантизирует LTX 2.3 → его 44 DiT-блока
  переименовываются из `model.diffusion_model.layers.*` (на которые нацелен regex
  `pollockjj/ComfyUI-MultiGPU` DisTorch2) в `model.diffusion_model.transformer_blocks.*`.
  DisTorch2 не находит этих блоков, тихо сваливает все 17 ГБ DiT на `cuda:0`
  одной карты и OOM-ит как только дело доходит до 720p-апскейла.
- **Решение**: hand-roll split в `core/gguf_split.py` (44 блока раскладываются по
  картам в соответствии со стратегией), forward-hook установлен для передачи
  скрытых состояний между картами между шагами KSampler. Никакого offload-to-CPU,
  никакого silent fallback.

---

## License этого CHANGELOG

`CHANGELOG.md` распространяется под **GPL-3.0-or-later**, как и код этого проекта.
