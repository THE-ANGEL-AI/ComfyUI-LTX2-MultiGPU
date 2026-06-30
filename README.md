# LTX-2 MultiGPU — гибридная загрузка на 2 видеокарты для LTX 2.3

> Запускайте **LTX 2.3 22B GGUF** на двух видеокартах без нехватки памяти. Сделано для Kaggle T4×2 (2 × 15 ГБ), проверено также на RTX 4090 / A5000.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Sponsor: Boosty](https://img.shields.io/badge/Sponsor-Boosty-orange.svg)](https://boosty.to/the_angel/donate)
[![ComfyUI Custom Node](https://img.shields.io/badge/ComfyUI-Custom_Node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-0.3.0--pre-green.svg)]()

> 🚧 **Скоро будет демо-GIF.** Положите файл `docs/hero.gif` (480p→720p прогон) в этот репозиторий, чтобы заменить этот блок. А пока просто добавьте ноду **LTX-2 Memory Diagnostics** перед KSampler — она покажет состояние обеих видеокарт в начале сессии.

---

## Что это и зачем — простыми словами

**LTX 2.3** — это нейросеть для генерации видео. Версия 22B (на 22 миллиарда параметров) занимает **~20 ГБ** в видеопамяти и **не помещается** в одну карту на 15 ГБ. Если попробовать загрузить её целиком на одну карту — получите ошибку `OOM (Out of Memory)`, то есть «память закончилась».

В этом пакете мы **делим** тяжёлую модель на две части и кладём каждую часть на свою видеокарту. Обе карты работают вместе, передавая друг другу промежуточные данные через специальный механизм (forward hook). В итоге:

- видео генерируется быстрее,
- памяти хватает,
- карты не простаивают.

Никаких танцев с бубном, никакого перекладывания модели в оперативку процессора между шагами — всё лежит в VRAM и работает параллельно.

---

## Если хотите поддержать проект 💚

Проект бесплатный и без рекламы, но разработка и тестирование требуют времени и GPU-часов на Kaggle/Colab. Если вам зашло и хочется сказать «спасибо»:

👉 **[Поддержать на Boosty](https://boosty.to/the_angel/donate)** — там можно перевести любую сумму даже анонимно.

Куда идут деньги:

- оплата Kaggle/Colab GPU-часов для тестирования на разных конфигурациях,
- новые квантизации LTX-GGUF (Q3, Q2),
- эксперименты с профилями для нестандартного железа (RTX 3090 + T4, A5000 + 3060 и т.д.),
- документация и переводы.

---

## Что в коробке

Шесть нод появятся в разделе **Add Node → LTX-2 MultiGPU**:

| Технический ключ (для workflow_api.json) | Отображение в меню | Что делает простыми словами | Чем заменяет |
|---|---|---|---|
| `LTX2_MultiGPU_HybridSplitLoader` | **Разделитель модели (2 GPU)** | Загружает GGUF DiT, делит 44 блока между двумя картами, соединяет их хуком для передачи данных | `UnetLoaderGGUFDisTorch2MultiGPU` |
| `LTX2_MultiGPU_GemmaHybridLoader` | **Загрузчик промптов (Gemma 3)** | Загружает Gemma 3 12B FP4 + `text_projection` как один CLIP, кладёт на нужные карты | `DualCLIPLoaderDisTorch2MultiGPU` |
| `LTX2_MultiGPU_MemoryDiagnostics` | **Диагностика видеопамяти** | Перед запуском считает VRAM/RAM бюджет, авто-выбирает стратегию, читает реальный quant из GGUF header | — |
| `LTX2_MultiGPU_DeviceStrategy` | **Переключатель стратегии** | Позволяет переключить стратегию прямо во время сессии, без перезагрузки модели | — |
| `LTX2_MultiGPU_VRAMParking` | **Парковка видеопамяти** | Временно убирает DiT блоки на CPU между Pass 1 и Pass 2, освобождая VRAM для VAE/upscale | — |
| `LTX2_MultiGPU_SageAttention` | **Патч SageAttention (T4)** | Ускоряет внимание ~1.5× через квантованное внимание (INT8 QK^T + FP16 PV) для Turing SM75 | — |

> 💡 **Привязка в workflow:** названия в workflow_api.json не менялись — по-прежнему левый столбец из таблицы. Русские слова из второго столбца нужны только чтобы быстро найти ноду в меню глазами.
>
> 🧪 **Готовые воркфлоу:** лежат в папке [`example_workflows/`](example_workflows/) — полный 2-pass пайплайн, strategy switch demo, diagnostics-first.

---

## Как память разложена на картах (UD Q4_K_M + Gemma 12B FP4, Kaggle T4×2)

```
            ┌─────────── cuda:0 (14.5 ГБ) ───────────┐  ┌─────────── cuda:1 (14.5 ГБ) ───────────┐
            │  DiT блоки 0..21   (~7.0 ГБ)           │  │  DiT блоки 22..43  (~7.0 ГБ)           │
            │  text_projection   (~2.1 ГБ)           │  │  Gemma 3 12B FP4    (~7.5 ГБ)          │
            │  Video VAE         (~1.4 ГБ)           │  │  Audio VAE          (~0.4 ГБ)          │
            │  Latent Upscaler   (~1.0 ГБ)           │  │                                       │
            │  LoRы (объединённые) (~1–3 ГБ)          │  │                                       │
            │  Scratch SageAttn  (~2.1 ГБ)           │  │  Scratch SageAttn   (~2.1 ГБ)          │
            │  ─────────────────────────────         │  │  ──────────────────────────────        │
            │  ≈ 12–15 ГБ итог   ⚠️ tight           │  │  ≈ 14–17 ГБ итог   ⚠️ tight            │
            └─────────────────────────────────────┘  └─────────────────────────────────────────┘
```

С UD Q4_K_M — влезает, но впритык. **Рекомендация:** используйте `VRAMParking` между Pass 1 и Pass 2 чтобы освободить DiT блоки на время VAE decode → upscale → VAE encode.

*(Цифры для **UD Q4_K_M** (~14 GB quant). Для Q5_K_M (~18 GB) — обязателен `blocks_30_70` + `VRAMParking`. Для Q3_K_XL (~12.5 GB) — запас с комфортом, можно `blocks_50_50`.)*

---

## Установка

Это занимает ~30 секунд.

**Windows (PowerShell):**

```powershell
cd D:\ComfyUI\custom_nodes
git clone https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU.git
cd ComfyUI-LTX2-MultiGPU
pip install -r requirements.txt
# Перезапустите ComfyUI. Четыре ноды появятся в "Add Node → LTX-2 MultiGPU".
```

**Linux / Kaggle / Colab:**

```bash
cd /workspace/ComfyUI/custom_nodes        # или где у вас ComfyUI
git clone https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU.git
cd ComfyUI-LTX2-MultiGPU
pip install -r requirements.txt
```

`requirements.txt`: `gguf`, `safetensors`. PyTorch и сам ComfyUI уже установлены — не ставьте их второй раз.

> 🛡 **Если у вас уже установлен другой пакет с тем же именем папки** — папка в `custom_nodes/` называется одинаково (``ComfyUI-LTX2-MultiGPU/``), и ComfyUI Manager подцепит *чужое* авторство, даже если поставить наш пакет рядом. **Удалите старую папку — один раз, до клона**:
> ```bash
> # Linux/Kaggle/Colab
> rm -rf /path/to/ComfyUI/custom_nodes/ComfyUI-LTX2-MultiGPU
> # Windows (PowerShell)
> Remove-Item -Recurse -Force D:\ComfyUI\custom_nodes\ComfyUI-LTX2-MultiGPU
> ```
> После этого делайте `git clone https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU.git` как обычно. Атрибуция будет корректной: **The Angel Studio / THE-ANGEL-AI**.

---

## Быстрый старт

1. **Положите модели** туда, где ComfyUI их ждёт:

   ```
   models/diffusion_models/ltx-2.3-Q6_K.gguf
   models/text_encoders/gemma-3-12b-it-FP4.safetensors
   models/text_encoders/gemma-3-12b-text_projection.safetensors
   ```

2. **Замените два ваших лоадера** — вместо `UnetLoaderGGUFDisTorch2MultiGPU` поставьте **LTX-2 Hybrid Split Loader**, а вместо `DualCLIPLoaderDisTorch2MultiGPU` — **LTX-2 Gemma Hybrid Loader**.

3. **Подключите LTX-2 Memory Diagnostics** прямо перед KSampler. Нода выведет в консоль снимок `nvidia-smi`, и вы убедитесь, что обе карты активны.

4. **Нажмите Generate.** Обе видеокарты должны загрузиться; первый прогон займёт ~20–40 с (раскладка блоков + установка forward-хука), потом скорость пойдёт нормальная.

### Выбор стратегии

В **LTX-2 Hybrid Split Loader** есть выпадающее меню `split_strategy`:

| Стратегия | Как раскладывает DiT | Когда выбирать |
|---|---|---|
| `blocks_50_50` *(по умолчанию)* | 22 блока на каждой карте | Сбалансированное 2 × 15 ГБ железо |
| `blocks_30_70` | 13 блоков на cuda:0, 30 на cuda:1 | На cuda:0 ещё живут VAE + Upscaler + тяжёлые LoRы |
| `pipeline` | Весь DiT на cuda:1, Gemma на cuda:0 | На cuda:0 навешаны ControlNet / доп. энкодеры |
| `single_cuda0` | Всё на cuda:0 | Фолбэк на одну карту / отладка |
| `single_cuda1` | Всё на cuda:1 | Фолбэк на одну карту / отладка |

Хотите сменить стратегию на лету? Подключите **LTX-2 Device Strategy Switch** между лоадером и KSampler.

В **LTX-2 Hybrid Split Loader** и **LTX-2 Device Strategy Switch** доступен виджет `donor_device`:

| donог_device | Семантика |
|---|---|
| `auto` *(по умолчанию)* | Secondary device из ComfyUI (обычно cuda:1 в dual-GPU). |
| `cuda:0` | Явно — **primary** карта (даже для pipeline/single_cuda1, где DiT идёт на secondary). |
| `cuda:1` | Явно — **secondary** карта (override авто-выбора). |

Зачем это нужно: если у вас асимметричное железо (RTX 3090 + RTX 3060) или специфическая раскладка, где вы хотите вручную задать, на какой карте лежит вторая половина DiT. В `auto`-режиме (по умолчанию) всё работает без ручных tweak'ов.

---

## Почему так получилось? (техническая часть)

Существующая нода `pollockjj/ComfyUI-MultiGPU` ищет блоки DiT регуляркой `model.diffusion_model.layers.*`. После того, как `city96/ComfyUI-GGUF` расквантизирует GGUF в fp16, **44 блока LTX-Video превращаются в имена `model.diffusion_model.transformer_blocks.*`** — это совсем другой префикс. DisTorch2 не находит ни одного блока, тихо сваливает все 17 ГБ DiT на cuda:0, и вылетает с OOM, как только дело доходит до апскейла 720p.

**Починка**: не пытаемся патчить DisTorch2 — мы сами грузим GGUF, сами раскладываем каждый тензор по картам, сами пишем cross-card forward-hook.

---

## Железо

**Проверено и работает:**

- **2 × NVIDIA T4 15 ГБ** *(Kaggle по умолчанию — главная цель)*
- **2 × RTX 4090 24 ГБ**
- **RTX A5000 24 ГБ** в паре с **RTX 3090 24 ГБ** *(асимметрия? выбирайте `blocks_30_70`)*

**Минимум:** 2 × 15 ГБ. Фолбэк на одну карту (`single_cuda0` / `single_cuda1`) требует 24+ ГБ.

**CPU offload для DiT не поддерживается.** Гонять ~17 ГБ через PCIe на каждом шаге KSampler стоило бы ~60 с на генерацию — DiT остаётся в VRAM. (С Gemma другая история — см. совет ниже.)

---

## Советы и грабли

### Освободите ~9.6 ГБ между картами

Включите `eject_models = True` в **LTX-2 Gemma Hybrid Loader**. После загрузки и проектор энкодера (~7.5 ГБ на cuda:1), и `text_projection` (~2.1 ГБ на cuda:0) уйдут в CPU одним вызовом — при условии, что ваша следующая нода дёрнет `model_unload()`.

⚠️ Пере-кодирование промпта в той же сессии не сработает — сэмплер не сможет перезагрузить `text_projection`, пока другая модель держит GPU. Используйте `eject_models` только для **single-pass encoding** воркфлоу.

### Длинные промпты не лезут в 7.5 ГБ на cuda:1?

KV-кеш Gemma растёт с длиной промпта. Если получаете OOM на cuda:1 — переключитесь на `pipeline`: тогда весь энкодер уйдёт на cuda:0, а cuda:1 освободится под скрытые состояния DiT.

### 720p-апскейл всё равно OOM-ит?

Попробуйте сначала `blocks_30_70` — он чуть сдвигает DiT в сторону cuda:1, освобождая cuda:0 для момента апскейла. Всё ещё тесно? Вставьте тайловый VAE между KSampler и апскейлером; cross-GPU тайловое внимание пока экспериментальное.

### Где смотреть `nvidia-smi` во время генерации?

Подключите **LTX-2 Memory Diagnostics** перед KSampler. Нода печатает в консоль снапшот обеих карт: занятая VRAM и загрузка GPU в начале каждого прогона.

---

## Проверенные GGUF-квантизации (Unsloth Dynamic 2.0)

> **Unsloth Dynamic 2.0** — per-layer mixed-precision: attention слои в Q6/Q8, FFN в Q4/Q3. Качество выше чем static imatrix quant при том же размере.

| GGUF-квантизация | Размер | Bits/param | 2×T4? | Рекомендация |
|---|---|---|---|---|
| **UD-Q4_K_M** | ~14 GB | ~4.2 | ✅ OK | **Рекомендовано** — sweet spot качество/память |
| UD-Q3_K_XL | ~12.5 GB | ~3.5 | ✅ OK | Запасной вариант, 1×T4 совместим |
| UD-Q5_K_M | ~18 GB | ~5.2 | ⚠️ Tight | Только с blocks_30_70 + VRAMParking |
| UD-Q2_K_XL | ~11 GB | ~2.6 | ✅ OK | Экономия памяти, заметная потеря качества |
| UD-Q6_K | ~20 GB | ~6.5 | ❌ OOM | Только на 24 GB картах |

> 💡 **Kaggle T4×2 (14.5 GB каждая):** UD-Q4_K_M — оптимальный выбор. Q5_K_M возможен с `blocks_30_70` + VRAMParking. Запустите [`ltx2_diagnostics_first.json`](example_workflows/ltx2_diagnostics_first.json) чтобы проверить ваш quant перед загрузкой.

---

## Лицензия и юридические моменты

**Веса модели LTX 2.3** распространяются под **LTX-Video Community License** — коммерческое использование ограничено. Перепроверьте условия лицензии на Hugging Face-страничке модели, прежде чем выпускать что-либо на основе этого пакета.

**Код пакета распространяется под GPL-3.0-or-later** (см. файл `LICENSE`). Сами веса модели — под LTX-Video Community License / Gemma License, как указано выше.

---

## Документы проекта

Подробная документация проекта лежит в папке [`docs/`](docs/):

* [**CHANGELOG.md**](docs/CHANGELOG.md) — история изменений по версиям (Keep-a-Changelog формат, semver).
* [**SECURITY.md**](docs/SECURITY.md) — полиция безопасности: как сообщать об уязвимостях, supported-versions, response-timeline, GPL-3.0 downstream-copyleft notes.
* [**CONTRIBUTING.md**](docs/CONTRIBUTING.md) — для контрибьюторов: fork/branch/PR, Conventional Commits, DCO sign-off, code-style.

Все три файла распознаются GitHub'ом автоматически (см. <https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/creating-a-default-community-health-file>).

--

## Хочется поковыряться

Самое интересное живёт в этих файлах:

- `core/gguf_split.py` — GGUF-сплиттер + установка forward-хука + apply_strategy,
- `core/memory_tracker.py` — Kaggle Edition: quant-aware VRAM/RAM расчёт + auto-strategy,
- `core/vram_parking.py` — парковка DiT блоков на CPU между проходами,
- `core/sage_attention.py` — интеграция SageAttention-SM75 для T4.

Если хотите подробную дизайн-историю (размеры компонентов, математика per-strategy, почему `cpu` отвергается для DiT) — см. `PLAN.md` и `MODEL_FACTS.md` в родительской директории.

**PR приветствуются**, особенно по темам:

- новые GGUF-квантизации,
- профили для нестандартного / асимметричного железа,
- скорость (время шага, узкие места PCIe).

---

## Поддержать проект (повтор)

👉 **[https://boosty.to/the_angel/donate](https://boosty.to/the_angel/donate)** — даже небольшая сумма помогает.

---

## Участники

Автор: **The Angel Studio** ([@THE-ANGEL-AI](https://github.com/THE-ANGEL-AI))  
Лицензия кода: **GPL-3.0-or-later**. Веса LTX 2.3 — **LTX-Video Community License** (см. раздел выше).

Стоит на плечах:

- [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — деквантизация GGUF (используем как upstream GGUF-loader),
- [pollockjj/ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) — паттерн, который мы заменили (DisTorch2 не подходит для LTX-Video).
