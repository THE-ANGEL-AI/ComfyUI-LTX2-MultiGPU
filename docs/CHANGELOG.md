# История изменений (Changelog)

Все значимые изменения в этом проекте документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
и этот проект придерживается [Semantic Versioning](https://semver.org/lang/ru/).

---

## [v0.3.0-pre] — 2026-06-30 (Kaggle Edition: VRAM parking, SageAttention, VAE Loader, rebrand)

### Added (новые фичи)

- **VRAM Parking (White Paper §8.1)** — нода `LTX2_MultiGPU_VRAMParking` и модуль
  `core/vram_parking.py`. Временно переносит ВСЕ DiT блоки (+ embed/head слои) на
  CPU между Pass 1 и Pass 2, освобождая VRAM для VAE decode → upscale → VAE
  encode. Идемпотентна (флаг `parked` в `model_options`). Безопасна для GGUF —
  использует `_ltx2_original_to` вместо `inner.to()` чтобы не триггерить dequant.
  `unpark_dit()` делегирует в `apply_strategy()` для точного восстановления
  GPU-layout с хуками.

- **SageAttention-SM75 (White Paper §6)** — нода `LTX2_MultiGPU_SageAttention`
  и модуль `core/sage_attention.py`. Интеграция через `model_options['attention_patch']`
  — стандартный ComfyUI механизм. Автоопределение: `import sageattn` в рантайме;
  если не установлен — тихо возвращает модель без патча. Wrapper делегирует в
  `sageattn.sageattn(q, k, v)` для INT8 QK^T + FP16 PV на T4 (Turing SM75).
  `model.clone()` изолирует мутации от оригинального графа.

- **Kaggle Edition memory tracker** — полный реврайт `core/memory_tracker.py`:
  - `gguf_quant_aware_bytes(path)` — читает GGUF header (без загрузки весов!),
    считает реальный quant-размер по `QUANT_BITS_APPROX` (24 quant-типа: Q4_0..BF16),
    возвращает `(file_size, vram_quant, fp16_equiv)`.
  - RAM бюджет: `KAGGLE_SYSTEM_RAM_GB = 29.0`, `KAGGLE_VRAM_PER_T4_GB = 14.5`,
    `KAGGLE_VRAM_RESERVED_GB = 1.0` (резерв под драйверы), `PYTHON_COMFY_RAM_OVERHEAD_GB = 3.5`.
  - `auto_select_strategy()` — приоритет: blocks_50_50 > blocks_30_70 > pipeline.
  - Quant-предупреждения: если Q5_K_M не влезает → советует Q4_K_M или Q3_K_XL.
  - Совместимость: legacy `gguf_estimate_bytes()` сохранён.

- **example_workflows/** — 3 готовых воркфлоу JSON для импорта в ComfyUI:
  - `ltx2_full_2pass_video.json` — полный 2-pass пайплайн со всеми 7 нодами.
  - `ltx2_strategy_switch.json` — демо горячего переключения стратегий.
  - `ltx2_diagnostics_first.json` — pre-flight сравнение 4 quant-типов.

- **VAE Loader с выбором GPU (node 7/7)** — нода `LTX2_MultiGPU_VAELoader`
  (`🖼️ VAE Загрузчик (GPU)`). Загружает VAE через `comfy.sd.load_vae()` и
  размещает `first_stage_model` на выбранном GPU (`donor_device`:
  auto/cuda:0/cuda:1/cpu). Совместим с VAE Decode/Encode — возвращает
  `("VAE",)`. Позволяет разгрузить cuda:0 от VAE во время decode/encode
  на T4×2 (14.5 GB каждая).

- **ComfyUI-Manager PR** — подана заявка на регистрацию в центральной базе
  `custom-node-list.json` (PR #3037 в `Comfy-Org/ComfyUI-Manager`).
  После merge: автор = THE-ANGEL-AI, click-through URL = наш репо,
  `dreamfast` навсегда вытеснен из Manager UI.

### Changed (ребрендинг + UX)

- **CATEGORY → `"THE-ANGEL-AI"`** (все 7 нод). Было `"THE-ANGEL-AI / LTX-2 MultiGPU"` —
  дублирование бренда в subcategory создавало шум в ComfyUI Add Node меню.
  Теперь чистое уникальное имя.

- **DISPLAY_NAME с эмодзи** (все 7 нод). Современные названия с иконками:
  🔀 Разделитель DiT (2×GPU), 📝 Dual CLIP Загрузчик (Gemma 3),
  🩺 Диагностика VRAM, ⚙️ Стратегия GPU (hot-switch),
  🅿️ Парковка DiT (VRAM↔CPU), ⚡ SageAttention (T4 турбо),
  🖼️ VAE Загрузчик (GPU).

- **GemmaHybridLoader → Dual CLIP** — `DISPLAY_NAME` изменён с
  `"Загрузчик промптов (Gemma 3)"` на `"📝 Dual CLIP Загрузчик (Gemma 3)"`
  по запросу пользователя («не загрузчик промптов, а Dual CLIP / текстовый
  энкодер»).

- **pyproject.toml**: добавлен `Icon = "assets/icon.png"` (128×128, brand color
  `#3B379E`). Иконка отображается в ComfyUI Manager.

- **__init__.py docstring**: обновлён под 7 нод и новую CATEGORY.

### Fixed (dropdown-баг)

- **FIX dropdown-bug (nodes.py)**: `if choices:` guard в `INPUT_TYPES` срезал
  dropdown-формат `(choices,)` когда `folder_paths.get_filename_list()`
  возвращал пустой список — ComfyUI показывал голое текстовое поле
  `("STRING", ...)` вместо выпадающего списка. Затронуты 3 ноды:
  HybridSplitLoader (`unet_name`), GemmaHybridLoader (`clip_name1` +
  `projection_name`), MemoryDiagnostics (`unet_name` + `gemma_name`).
  Теперь dropdown рендерится всегда (даже пустой).

- **FIX (GemmaHybridLoader/MemoryDiagnostics)**: file-picker'ы для Gemma
  (.safetensors) теперь объединяют файлы из **обеих** папок —
  `text_encoders/` И `clip/` — через `dict.fromkeys()` dedup.
  Раньше каждый виджет смотрел только в одну папку.

### Fixed (критические исправления)

- **CRITICAL (`__init__.py`)**: `_build_config()` использовал
  `importlib.import_module('nodes')` который импортировал **ComfyUI root `nodes.py`**
  вместо нашего `nodes.py` → `NODE_CLASS_MAPPINGS` был пуст → все 6 нод были
  невидимы в ComfyUI. Заменён на относительный импорт `from . import nodes`.

### Changed (изменения)

- **nodes.py**: добавлены ноды 5 (`VRAMParking`), 6 (`SageAttention`),
  7 (`VAELoader`). Всего 7 нод в `NODE_CLASS_MAPPINGS`.
- **pyproject.toml**: `author="THE-ANGEL-AI"`, `PublisherId="THE-ANGEL-AI"`,
  `Icon="assets/icon.png"`, `reference="https://github.com/THE-ANGEL-AI/..."`,
  `DisplayName="LTX-2 MultiGPU"`.
- **README.md**: миграционная заметка для пользователей dreamfast.

### Test coverage

- `tests/test_vram_parking.py` — 11 тестов (идемпотентность, round-trip, missing config).
- `tests/test_sage_attention.py` — 9 тестов (мокинг `sys.modules`, делегирование wrapper).
- `tests/test_init.py` / `tests/test_nodes.py` — обновлены под 7 нод
  (+ VAE Loader контракт, + TestVAELoaderWidgets, CATEGORY/эмодзи/DISPLAY_NAME).
- Все тесты (6 модулей, 119 тестов) проходят: `python -m unittest discover tests -v`.

---

## [v0.2.2-pre] — 2026-06-30 (pre-T4x2 deploy: attribution + Russian UX)

### Проблема, которую решает этот релиз

Пользователи в ComfyUI Manager видели атрибуцию чужого проекта (`dreamfast`) и
только английские длинные названия нод в меню Add Node → root → непонятно, какие
имена для workflow-линковки и как добавлять узлы. Этот релиз чётко ставит
атрибуцию (THE-ANGEL-AI / The Angel Studio / `gi.the.angel@gmail.com`) и делает
display-имена нод понятными русскоязычным пользователям.

### Fixed (атрибуция для ComfyUI Manager)

- **FIX ATTRIB-1 (`__init__.py`)**: добавлены все required/optional metadata-поля,
  которые ComfyUI (включая ComfyUI Manager's registry) использует для attribution:
  - ``__version__ = "0.2.2-pre"`` (синхронизирован с pyproject.toml).
  - ``__author__ = "The Angel Studio"`` (был уже).
  - ``__author_email__ = "gi.the.angel@gmail.com"`` (NEW).
  - ``__author_github__ = "THE-ANGEL-AI"`` (NEW — ComfyUI Manager парсит это
    поле из некоторых forks и сопоставляет с GitHub API для верификации).
  - ``__repo__ = "https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU"`` (NEW
    — fallback для consumers, которые читают ``__repo__`` напрямую).
  - Большой header в module docstring с ascii-art-attribution banner: автор,
    repo, sponsor, license, 4 ноды одной строкой, явное
    **«НЕ форк каких-либо из тех проектов»** (anti-confusion guard против
    `dreamfast/ComfyUI-LTX2-MultiGPU` — у нас другой проект).
- **FIX ATTRIB-2 (`pyproject.toml`)**: ``[project].description`` теперь явно
  указывает авторство: *"Hybrid Multi-GPU split loader for LTX 2.3 GGUF on
  2×T4. Made by THE-ANGEL-AI. Не fork dreamfast."* (первая фраза осталась
  функциональной, второй абзац — attribution-guard).
- **FIX ATTRIB-3 (`pyproject.toml`)**: ``version`` bumped ``"0.2.1"`` → ``"0.2.2-pre"``.
- **FIX ATTRIB-4 (`__init__.py`)**: консольный banner с author/repo/version
  **opt-in** через env-var ``LTX2_MULTIGPU_VERBOSE=1``. По умолчанию — тихо,
  чтобы не засорять stdout ComfyUI при штатной загрузке. Stdout-fallback
  (`try: print(...) except Exception: pass`) на случай frozen-exe / systemd env.

### Fixed (Russian display names + clean menu grouping)

- **FIX UX-1 (`nodes.py`)**: 4 DISPLAY_NAME'а переведены на **русский** для
  человекочитаемости в меню Add Node. Технические class-keys (``NODE_ID``)
  и ``model_class`` values в ``NODE_CLASS_MAPPINGS`` НЕ переименованы — все
  существующие workflow_api.json остаются совместимыми. Финальные имена:
  - ``LTX2_MultiGPU_HybridSplitLoader`` → ``"Разделитель модели (2 GPU)"``
  - ``LTX2_MultiGPU_GemmaHybridLoader`` → ``"Загрузчик промптов (Gemma 3)"``
  - ``LTX2_MultiGPU_MemoryDiagnostics`` → ``"Диагностика видеопамяти"``
  - ``LTX2_MultiGPU_DeviceStrategy`` → ``"Переключатель стратегии"``
  В ComfyUI Add Node меню теперь: **Add Node → LTX-2 MultiGPU →**
  четыре русских имени в алфавитном порядке (Д < З < П < П). CATEGORY
  ``"LTX-2 MultiGPU"`` обеспечивает group-prefix auto-add в UI, поэтому в
  DISPLAY_NAME мы НЕ дублируем бренд-префикс (reviewer-revised style).
- **FIX UX-2 (`nodes.py`)**: каждый DISPLAY_NAME прокомментирован:
  *"Russian display name for users (grouped by CATEGORY). CATEGORY prefix
  adds the brand tag automatically in ComfyUI's Add Node menu, NODE_ID
  (technical class key) preserved for workflow_api/script compat."*

### Compatibility note

- ``NODE_CLASS_MAPPINGS`` keys (technical) **не изменены**: все 4 ключа —
  ``LTX2_MultiGPU_*`` — те же что в v0.2.1. Existing workflow_api.json
  старых версий остаются load-compatible.
- ``NODE_DISPLAY_NAME_MAPPINGS`` values изменены — это OK, поскольку эти
  values используются только для UI-render. Если какой-то downstream
  serailizes workflow_api.json по display-name (не NODE_ID), они увидят
  русские имена после upgrade; это правильный путь для v0.2.2-pre.
- ``__version__ = "0.2.2-pre"`` — pre-release marker; users на ``pip install
  --pre`` получат эту версию, на stable pin получат v0.2.1 пока не выйдет
  v0.2.2 stable.

---

## [v0.2.1] — 2026-06-30 (audit-driven bugfixes)

### Fixed (аудит post-v0.2.0: regressive correction + strategy-switch reliability)

- **FIXED (regression)**: ``_split_blocks_indices("blocks_30_70")`` теперь возвращает ``(13,)`` (13 блоков @ primary, 31 @ donor = честные 30/70). Раньше возвращал ``(14,)`` → 32/68 split (14 @ primary, 30 @ donor), что **противоречило имени стратегии** и обманывало ``memory_tracker.estimate_vram_budget``: heuristic проецировал бюджет для 30/70, runtime получал 32/68 и OOM-ил на edge-budget сценариях. Docstring в ``_split_blocks_indices`` синхронизирован под новое `(13,)`.
- **FIXED (apply_strategy блокировка)**: ``apply_strategy`` теперь использует ``inner._ltx2_original_to`` (unpatched ``nn.Module.to``) вместо ``inner.to`` для whole-model перемещений. Причина: ``_lock_inner_to`` (Risk #7 fix) monkey-патчит ``inner.to`` в no-op для device-moves, чтобы ComfyUI sampler не драгал split-блоки обратно на cuda:0. До этого fix'а -- первый вызов ``apply_strategy`` после ``hybrid_split_gguf`` НИЧЕГО не двигал (патч поглоцал вызов) и strategy-switch silently no-op'ил. Fallback ``getattr(_, ..., inner.to)`` -- применимо для случая ``apply_strategy`` до ``hybrid_split_gguf``.
- **FIXED (apply_strategy embed/head drift)**: при переключении ``whole-model`` → ``blocks_*`` стратегии через ``apply_strategy`` теперь вызывается ``_move_modules_with_prefix(diffusion, primary_dev, *EMBED_AND_HEAD_REL)`` после перемещения blocks 0..split_idx. Без этого ``time_embed`` / ``adaln`` / ``proj_in`` / ``proj_out`` оставались на cuda:1 из прошлой whole-model стратегии → device mismatch при sampling → runtime crash. Зеркалит ту же логику что в ``hybrid_split_gguf``.
- **FIXED (apply_strategy defensive fallback)**: новая ``else:``-ветка после ``target_dev``/split-mode блоков. Имитирует поведение ``hybrid_split_gguf``: если какой-то split-translate'ор окажется без соответствующей ветки в ``_split_blocks_indices`` (будущая ``blocks_70_30`` без branch), вместо silent no-op теперь UNCONDITIONAL ``[ComfyUI-LTX2-MultiGPU] WARN: apply_strategy: ...`` + ``whole-model move @ primary_dev`` -- ловится визуально + не разрушает model placement.
- **ADDED**: ``tests/test_gguf_split_blocks_indices.py`` -- regression-gate stdlib-unittest (без pytest). 10 методов: ``blocks_50_50``/``blocks_30_70`` (с share-percentage assertions) / ``pipeline``/``single_cuda0/1`` → пустой split; defensive unknown-strategy subtest loop; ``all_STRATEGIES_resolvable`` с explicit caveat про false-negative trap; STRATEGIES-tuple integrity assertion; ``LTX2_DIT_BLOCK_COUNT == 44`` invariant; ``_split_blocks_indices`` export sanity. Запуск: ``python -m unittest tests.test_gguf_split_blocks_indices -v``.
- **ADDED (apply_strategy docstring)**: новое ⚠️ **DiT-only** предупреждение на ``apply_strategy`` -- функция предполагает DiT-ModelPatcher от ``hybrid_split_gguf`` и НЕ предназначена для Gemma encoder patcher от ``load_gemma_hybrid`` (projections: text_projection@primary + encoder@donor делатсяя в ``load_gemma_hybrid``, не через hot-swap). Если в будущем понадобится apply_strategy для Gemma -- нужна отдельная функция с проекцией на Gemma layout.

### Compatibility note

- Никаких breaking changes API: ``_split_blocks_indices``, ``hybrid_split_gguf``, ``apply_strategy`` signatures не поменялись -- только contents/behavior. Downstream-форки, импортирующие ``from core.gguf_split import _split_blocks_indices``, полyчат корректное ``(13,)`` для ``blocks_30_70`` вместо buggy ``(14,)``.
- ``tests/`` -- новая директория. Без ``__init__.py``. Совместимо с stdlib unittest (run via ``python -m unittest tests.* -v``); pytest не требуется, но ``tests/__init__.py`` может потребоваться в CI когда pytest добавится в workflow (вне scope этого patch).

### Polish (post-audit: HIGH-1 + HIGH-2 + MED дополнения поверх регрессионного раунда выше)

#### HIGH severity

- **FIXED (HIGH-1, key unification)**: ``scripts/diagnostic.py`` и ``core/memory_tracker.py`` используют разные импна для одного и того же ключа footprint dict (``sage_scratch`` vs ``sage_attention_scratch``). Это silent-bug: components dict передавался через pipeline но ``.get()`` возвращал 0 для отсутствующего ключа → projection занижала VRAM на **~4.2 GB** (2.1 GB × 2 карты). Canonical key теперь ``sage_attention_scratch`` в обоих файлах; backward-compat fallback ``.get("sage_attention_scratch", components.get("sage_scratch", 0.0))`` kept в ``diagnostic.py:_compute_other`` для старых callers. test: ``tests/test_memory_tracker.py::TestV021PolishGuards::test_sage_attention_scratch_key_present``.
- **FIXED (HIGH-2, silent dtype drop в `_lock_inner_to._no_op_to`)**: Risk #7 fix из v0.2.1 ранеешеl перехватывал **весь** вызов ``inner.to(...)`` если видел device-like arg — тихо проглатывал сопутствующие ``dtype=torch.float16`` / ``memory_format=torch.channels_last`` / ``non_blocking=True``. KSampler иногда зовёт ``.to(dtype=...)`` для optimization — старый fix дропал dtype-cast → sampler-uncacheable weight-bits и потенциально OOM после merge. Новые helpers: ``_is_device_arg(a)`` (top-level device-detector), ``_strip_device(args, kwargs)`` (отфильтровывает device args/kwargs). Теперь: если device-move detected, вызов передаётся в ``original_to(*cleaned_args, **cleaned_kwargs)`` с dtype/memory_format сохранёнными; pure non-device calls идут в original_to без изменений; pure device-call возвращает ``inner`` (no-op).

#### MED severity

- **ADDED (MED-3, donor_device в `apply_strategy`)**: Node ``LTX2_MultiGPU_DeviceStrategy`` ранеешеl захардкодил secondary_dev для всех strategy-switch на лету, из-за чего user override из HybridSplitLoader (например `cuda:0` или `cpu`) молча игнорировался при переключении стратегии через DeviceStrategy ноду. Сигнатура apply_strategy теперь: ``apply_strategy(patcher, strategy, verbose=False, donor_device="auto")``. Degenerate-guard для ``cpu`` как donor (для DiT — anti-pattern → fallback на secondary_dev с UNCONDITIONAL WARN зеркалит hybrid_split_gguf). DeviceStrategy INPUT_TYPES получил required widget ``donor_device ∈ {auto, cuda:0, cuda:1}`` (default ``auto``). ComfyUI auto-fills default при отсутствии в workflow_api.json → backward-compat OK.
- **ADDED (MED-3 follow-up, `model_options` rename)**: Key ``"secondary"`` в ``patcher.model_options["ltx2_multigpu_split"]`` после добавления `donor_device` widget стал misleading: значение могло быть primary или secondary. Canonical key теперь ``"effective_donor"``. Legacy keys ``"secondary"`` / ``"donor"`` оставлены для backward-compat с явным комментарием ``# legacy alias``.
- **FIXED (MED-4, loras_estimate в pipeline projection)**: ``memory_tracker.estimate_vram_budget._project("pipeline")`` ранеешеl не включал ``loras_estimate`` (2.55 GB) → projection silent-bug для workflow с LoRами + pipeline-стратегией (false OK → real OOM). Теперь loras учитывается на стороне, где лжит DiT (cuda:1 для pipeline/single_cuda1, cuda:0 для остальных). test: ``tests/test_memory_tracker.py::TestProjectPerStrategy::test_pipeline_includes_loras_on_cuda1``.

#### LOW / cosmetic

- **CHANGED (MED-5, version sync)**: ``pyproject.toml``, ``__init__.py``, README version-badge: ``0.2.0`` → ``0.2.1``. CHANGELOG теперь имеет две подсекции под v0.2.1 (audit выше + polish ниже) — release-history закрыта.
- **ADDED (TEST-extend)**: ``tests/test_memory_tracker.py`` — 4 unittest-класса с 18 тестами: TestV021PolishGuards (HIGH-1 guard), TestProjectPerStrategy (MED-4 math для каждой strategy), TestClassifyTensor (DiT tensor-name → category mapping), TestResolveDonorDevice (auto/cuda:0/cuda:1/cpu + edge cases). Stdlib-only, без pytest.

### Compatibility note (cumulative)

- **Backward-compat API**: ``_split_blocks_indices``, ``hybrid_split_gguf``, ``apply_strategy`` core signatures не сломаны (MED-3 added positional kwarg ``donor_device="auto"`` → optional).
- **Backward-compat УI**: DeviceStrategy `donor_device` widget теперь REQUIRED с default ``"auto"`` → старые workflow_api.json (без поля) auto-fill default, runtime OK.
- **Backward-compat metadata**: ``patcher.model_options["ltx2_multigpu_split"]["effective_donor"]`` — новый canonical. ``"secondary"``/``"donor"`` legacy keys сохранены. Consumers могут migrate to ``effective_donor`` без breaking old code.

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

- **ADDED**: `.github/FUNDING.yml` — нативная GitHub-кнопка Sponsor →
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

### Documentation (структура папок)

- **CHANGED**: переезд `SECURITY.md` / `CHANGELOG.md` / `CONTRIBUTING.md` из root в `docs/` (git mv с сохранением истории). `README.md` остаётся в root для GitHub landing-page visibility. Причина: чистая root-структура (только runtime metadata + код + .github/); `docs/` — расширяемая папка для future deep-docs (`ARCHITECTURE.md` / `DESIGN.md` / `API.md` / `TROUBLESHOOTING.md` — план).
- **ADDED**: `docs/index.md` — 11-строчный навигатор docs/: title `# Documentation (docs/)` + intro-блок + 3 bullets с one-line описанием и markdown-links (`./CHANGELOG.md` / `./SECURITY.md` / `./CONTRIBUTING.md`) + back-link `[**README.md в корне**](../README.md) + GH community-health-file URL citation + GPL-3.0 footer.
- **CHANGED**: cross-references в переехавших файлах обновлены под новые relative-path'ы: `docs/SECURITY.md` теперь ссылается на `./CHANGELOG.md` + `../LICENSE`; `docs/CONTRIBUTING.md` теперь ссылается на `./SECURITY.md` + `./CHANGELOG.md`; `docs/CHANGELOG.md`: wrong-prefix `submodule/.github/FUNDING.yml` → `.github/FUNDING.yml` (relic from superproject-era description, теперь относительный путь to submodule-root).
- **CHANGED**: root `README.md` получил новую секцию `## Документы проекта` (между `## Лицензия...` и `## Хочется поковыряться`) с markdown-links на 3 файла в `docs/`.

### Build (version sync package metadata)

- **CHANGED**: `pyproject.toml [project] version = "0.1.0"` → `"0.2.0"`. С комментарием выше строки: `# 2026-06-30 — synchronized with CHANGELOG.md v0.2.0 (SoloAngel/SP-release)`. Effect: `pip install --upgrade` теперь правильно детектит новую версию — раньше возвращал 'requirement already satisfied' из-за несинхронного metadata mismatch (pyproject был 0.1.0 а HEAD-коммит уже содержал CHANGELOG v0.2.0).
- **CHANGED**: `__init__.py module-level __version__ = "0.1.0"` → `"0.2.0"`. С комментарием `# VERSION синхронизирован с pyproject.toml (release v0.2.0, 2026-06-30)`. Runtime-variable, синхронизируется по manual-rule с pyproject.toml.
- **CHANGED**: root `README.md` version-badge URL: `version-0.1.0-green.svg` → `version-0.2.0-green.svg`. Visible visual cue for users browsing repo root.

### CI

- **ADDED** (`278f2ba`): `.github/workflows/ci.yml` (Smoke CI). Triggers: `push: branches: [main]`, `pull_request: branches: [main]`, `workflow_dispatch` (manual). Matrix: Python 3.10 / 3.11 / 3.12, `fail-fast: false` (один упавший Python не отменяет остальные). Steps: `actions/checkout@v4` + `actions/setup-python@v5` (с `cache: 'pip'` + `cache-dependency-path: requirements.txt,pyproject.toml`) + `pip install CPU-only torch (~200 МБ) + pip install -r requirements.txt` + `py_compile __init__.py nodes.py core/gguf_split.py core/gguf_reader.py core/memory_tracker.py` + `import-check` через `python -c "import sys; sys.path.insert(0, '.'); import nodes; print('OK')"`. Concurrency group `ci-smoke-${{ github.ref }}` + `cancel-in-progress: true` (экономит CI-minutes на force-push в PR'е). Permissions: `contents: read` (минимум).
- **Effect**: ImportError regressions теперь ловятся ДО merge в `main`. Раньше регрессия в `apply_strategy` или silent `import torch` failure проходила review без визуального smoke-test и обнаруживалась только через community issue-report.

### Fixed (audit trail)

- **FIXED** (`3fbf84b`, `74f08e54`): pre-existing typo `nnodes.py` → `nodes.py` в `docs/SECURITY.md` "Out of Scope" bullet. Источник: code-review pass во время docs-restructure (`74f08e54`). Также удалён redundant phrase "см. README «Советы и грабли» +" — ссылка на root README была prose-only описательная, новая формулировка стала точнее: `\`nodes.py\` § \`verbose_log\``.

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
