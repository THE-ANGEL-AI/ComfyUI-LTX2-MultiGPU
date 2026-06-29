# ComfyUI-LTX2-MultiGPU

> **Автор:** The Angel Studio  (`gi.the.angel@gmail.com`)
> **Maintainer:** [@THE-ANGEL-AI](https://github.com/THE-ANGEL-AI) (GitHub login) — display name: *The Angel Studio* — `<gi.the.angel@gmail.com>`
> **Цель:** Задействовать 2 GPU (rekomendуется 2×T4 15 ГБ на Kaggle) для генерации видео **LTX 2.3 22B GGUF**.
> **Стратегия:** Hybrid — DiT tensor-parallel 50/50 по блокам + pipeline (Gemma / Audio VAE на свободной карте).

## Зачем

`UnetLoaderGGUFDisTorch2MultiGPU` из `pollockjj/ComfyUI-MultiGPU` использует regex `model.diffusion_model.layers.X.*` для поиска блоков DiT. После dequant GGUF (city96) тензоры DiT LTX 2.3 имеют префикс **`transformer_blocks`** — DisTorch2 молча не находит слои и грузит весь DiT на `main_device` (cuda:0) → OOM при 720p.

Этот пакет **не патчит DisTorch2** — он грузит GGUF вручную и строит кастомный `nn.Module` с явным шардированием 44 блоков и forward-hook для передачи hidden state между GPU.

## Установка

```bash
cd D:\ComfyUI\custom_nodes
git clone https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU.git
cd ComfyUI-LTX2-MultiGPU
pip install -r requirements.txt
# Перезапустить ComfyUI
```

После `git clone` скопировать папку `ComfyUI-LTX2-MultiGPU` в `<ComfyUI>\custom_nodes\`
(либо работать прямо склонированно, сделав `<ComfyUI>\custom_nodes\ComfyUI-LTX2-MultiGPU`
симлинком или копией).

Зависимости: `gguf`, `safetensors` (torch и comfy не дублировать — они уже в ComfyUI).

## Ноды (появляются в Add Node menu в категории `LTX-2 MultiGPU`)

| NODE_CLASS_ID | Назначение | Заменяет |
|---|---|---|
| `LTX2_MultiGPU_HybridSplitLoader` | GGUF DiT split 50/50 + ModelPatcher | `UnetLoaderGGUFDisTorch2MultiGPU` |
| `LTX2_MultiGPU_GemmaHybridLoader` | жёсткая загрузка Gemma 12B FP4 на cuda:1 + `text_projection` на cuda:0 | `DualCLIPLoaderDisTorch2MultiGPU` |
| `LTX2_MultiGPU_MemoryDiagnostics` | pre-flight VRAM оценка + лог в консоль | — |
| `LTX2_MultiGPU_DeviceStrategy` | выбор стратегии (50/50, 30/70, pipeline, single) | — |

## Архитектура Hybrid Split

```
┌────────────────── cuda:0 ──────────────────┐  ┌────────────────── cuda:1 ──────────────────┐
│  blocks 0..21  ────────┐                  │  │                  ┌────── blocks 22..43     │
│  embed_tokens  ─┐      │  forward_hook     │  │  forward_hook    │                         │
│  time_embed    ─┤      │  (hidden.to(1)) ─┼──┼──────────────────┼─►  Gemma 12B FP4 (text) │
│  adaLN         ─┤      ▼                  │  │                  │      Gemma encoder      │
│  proj_out ─► ◄──┘      norm               │  │                  │      text_projection ──►│
│  Video VAE                                   │  │                  Audio VAE               │
│  Latent Upscaler                            │  │                                        │
│  LoRAs (merge)                              │  │                                        │
└──────────────────────────────────────────────┘  └────────────────────────────────────────┘
```

Ключевой приём: **`register_forward_hook`** на 21-м блоке передаёт `hidden_states.to('cuda:1')` для последующих 22 блоков; обратный hook на 44-м блоке возвращает результат на `cuda:0` для финальных слоёв.

## Поддерживаемые GPU

Минимум: 2× 15 ГБ (T4 на Kaggle).
Рекомендовано: 2× 24 ГБ (RTX 4090, A5000).

## Лицензия

MIT.

## Предупреждение

Проверять лицензии компонентов пайплайна LTX 2.3 на странице Hugging Face — коммерческое использование имеет ограничения согласно LTX License.
