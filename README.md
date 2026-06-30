# LTX-2 MultiGPU — Hybrid 2-GPU Split for LTX 2.3

> Run **LTX 2.3 22B GGUF** across 2 GPUs without OOM. Built for Kaggle T4×2 (2 × 15 GB), tested on RTX 4090 / A5000 too.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![ComfyUI Custom Node](https://img.shields.io/badge/ComfyUI-Custom_Node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)]()

> 🚧 **Demo GIF coming soon.** Drop a `docs/hero.gif` (480p→720p run) in this repo to replace this block. Until then, run **LTX-2 Memory Diagnostics** before your KSampler — it'll print both cards' VRAM snapshot at session start, which is the next best thing.

---

## TL;DR

You're trying to run **LTX 2.3 22B** (GGUF Q4_K_M / Q5_K_M / Q6_K) across **2 × 15 GB GPUs** — most likely a pair of T4s on Kaggle. The vanilla `UnetLoaderGGUFDisTorch2MultiGPU` OOMs at 720p because DisTorch2 can't find LTX-Video's DiT blocks (it greps for `model.diffusion_model.layers.*`, but LTX exposes them as `model.diffusion_model.transformer_blocks.*`).

This pack skips DisTorch2 entirely. It loads the GGUF itself, splits the 44 DiT blocks across your two GPUs, pins Gemma 3 12B FP4 to whichever card has room, and uses a forward hook to hand hidden-state between cards each step. No offload-to-CPU stalling, no 720p OOM.

---

## What's in the box

Four nodes land in **Add Node → LTX-2 MultiGPU**:

| Node | What it does | Replaces |
|---|---|---|
| **LTX-2 Hybrid Split Loader** | Loads GGUF DiT, splits 44 blocks across your 2 GPUs, wires the cross-card forward hook | `UnetLoaderGGUFDisTorch2MultiGPU` |
| **LTX-2 Gemma Hybrid Loader** | Loads Gemma 3 12B FP4 + `text_projection` as a single CLIP, pinned to the right cards | `DualCLIPLoaderDisTorch2MultiGPU` |
| **LTX-2 Memory Diagnostics** | Pre-flight VRAM check — projects your strategy fit, prints an `nvidia-smi` snapshot | — |
| **LTX-2 Device Strategy Switch** | Hot-swap the split strategy mid-session (re-routes forward hooks, no reload) | — |

---

## Target VRAM layout (Q6_K + Gemma 12B FP4)

```
            ┌─────────── cuda:0 (15 GB) ──────────┐  ┌─────────── cuda:1 (15 GB) ──────────┐
            │  DiT blocks  0..21   (~9 GB)         │  │  DiT blocks 22..43  (~9 GB)        │
            │  text_projection     (~2.1 GB)       │  │  Gemma 3 12B FP4    (~7.5 GB)      │
            │  Video VAE           (~1.4 GB)       │  │  Audio VAE          (~0.4 GB)      │
            │  Latent Upscaler     (~1.0 GB)       │  │                                    │
            │  LoRAs (merged)      (~1–3 GB)       │  │                                    │
            │  SageAttn scratch    (~2.1 GB)       │  │  SageAttn scratch    (~2.1 GB)     │
            │  ───────────────────────────────     │  │  ──────────────────────────────    │
            │  ≈ 10–14 GB total   ✅ fits          │  │  ≈ 16–17 GB total   ✅ fits        │
            └─────────────────────────────────────┘  └────────────────────────────────────┘
```

That leaves ~1–4 GB of activation headroom per card — enough for 720p upscale. *(Numbers include the **2.1 GB SageAttn scratch** per card; subtract 2.1 on each side if you're running SageAttn off.)*

---

## Install

About 30 seconds end-to-end.

**Windows (PowerShell):**

```powershell
cd D:\ComfyUI\custom_nodes
git clone https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU.git
cd ComfyUI-LTX2-MultiGPU
pip install -r requirements.txt
# Restart ComfyUI. The four nodes appear under "Add Node → LTX-2 MultiGPU".
```

**Linux / Kaggle / Colab:**

```bash
cd /workspace/ComfyUI/custom_nodes        # or wherever your ComfyUI lives
git clone https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU.git
cd ComfyUI-LTX2-MultiGPU
pip install -r requirements.txt
```

`requirements.txt`: `gguf`, `safetensors`. PyTorch and ComfyUI itself are already installed — don't double them up.

---

## Quick start

1. **Drop the models** where ComfyUI expects them:

   ```
   models/diffusion_models/ltx-2.3-Q6_K.gguf
   models/text_encoders/gemma-3-12b-it-FP4.safetensors
   models/text_encoders/gemma-3-12b-text_projection.safetensors
   ```

2. **Swap your two loaders** — replace `UnetLoaderGGUFDisTorch2MultiGPU` with **LTX-2 Hybrid Split Loader**, and replace `DualCLIPLoaderDisTorch2MultiGPU` with **LTX-2 Gemma Hybrid Loader**.

3. **Plug in LTX-2 Memory Diagnostics** right before your KSampler. It logs a `nvidia-smi` snapshot so you can confirm both cards are live.

4. **Hit Generate.** Both GPUs should ramp up; first generation warm-up takes ~20–40 s (block placement + forward-hook setup), then throughput per step is normal.

### Pick a split strategy

`LTX-2 Hybrid Split Loader` exposes a `split_strategy` dropdown:

| Strategy | DiT layout | Pick this when |
|---|---|---|
| `blocks_50_50` *(default)* | 22 blocks per card | Balanced 2 × 15 GB hardware |
| `blocks_30_70` | 13 blocks on cuda:0, 30 on cuda:1 | cuda:0 also hosts VAE + Upscaler + heavy LoRAs |
| `pipeline` | Whole DiT on cuda:1, Gemma on cuda:0 | You're stacking ControlNet / extra encoders on cuda:0 |
| `single_cuda0` | Everything on cuda:0 | Single-GPU fallback / debugging |
| `single_cuda1` | Everything on cuda:1 | Single-GPU fallback / debugging |

Want to swap mid-run? Wire `LTX-2 Device Strategy Switch` between the loader and KSampler.

---

## How did we get stuck?

`pollockjj/ComfyUI-MultiGPU` discovers DiT blocks with the regex `model.diffusion_model.layers.*`. After `city96/ComfyUI-GGUF` dequantizes the GGUF into fp16, **LTX-Video's 44 blocks show up as `model.diffusion_model.transformer_blocks.*`** — totally different prefix. DisTorch2 matches zero blocks, silently drops the entire 17 GB DiT on `cuda:0`, and OOMs the moment you reach 720p upscale.

Fix: don't pretend DisTorch2 can be patched — load the GGUF yourself, hand-place every tensor, and hand-roll the cross-card forward hook.

---

## Hardware

**Tested and confirmed working:**

- 2 × NVIDIA T4 15 GB *(Kaggle default — main target)*
- 2 × RTX 4090 24 GB
- RTX A5000 24 GB paired with RTX 3090 24 GB *(asymmetric? go `blocks_30_70`)*

**Minimum:** 2 × 15 GB. Single-GPU fallback (`single_cuda0` / `single_cuda1`) wants 24+ GB.

**CPU offload is not supported for DiT.** Shuttling ~17 GB across PCIe every KSampler step would cost ~60 s per generation — DiT stays pinned in VRAM. (Gemma is a different story — see the next tip.)

---

## Tips & gotchas

### Free ~9.6 GB between cards

Flip `eject_models = True` on **LTX-2 Gemma Hybrid Loader**. After load, both projects
encoder (~7.5 GB on cuda:1) **and** `text_projection` (~2.1 GB on cuda:0) flip to CPU in a single call, as long as your downstream node invokes `model_unload()`.

Heads up: re-conditioning in the same session won't work — sampler can't reload `text_projection` while another model holds the GPU. Use `eject_models` only for **single-pass encoding** workflows.

### Long prompts blow past 7.5 GB on cuda:1?

Gemma's KV cache grows with prompt length. If you find yourself OOMing on cuda:1, switch to `pipeline` — that moves the whole encoder to cuda:0, leaving cuda:1 dedicated to DiT hidden-state math.

### 720p upscale still OOMs?

Try `blocks_30_70` first — it shifts DiT weight slightly toward cuda:1, freeing cuda:0 for the upscale moment. Still tight? Pipe a tiled VAE between KSampler and the upscaler node; cross-GPU tiled attention is still experimental.

### Where do I see `nvidia-smi` while generating?

Drop an **LTX-2 Memory Diagnostics** node upstream of KSampler. It's a terminal-output node that prints a per-card VRAM + utilization snapshot at the start of every run.

---

## Verified GGUF quantizations

Sourced from `MODEL_FACTS.md §2` (component footprint table). For exact file sizes of the LTX-Video 22B GGUF release, check the model's Hugging Face card — these are approximate baseline sizes for a fresh download.

| GGUF quant | Approx. file size | Result |
|---|---|---|
| Q4_K_M | ~17 GB | Comfortable — fits with margin |
| Q5_K_M | ~21 GB | Fits — default for Kaggle T4×2 |
| Q6_K | ~24 GB | Best quality — works if both cards have ≥24 GB |
| Q3 / Q2 | < 12 GB | Untested, expect visible quality loss |

---

## Compatibility note

LTX 2.3 model weights are released under the **LTX-Video Community License** — commercial use has restrictions. Double-check the license terms on the model's Hugging Face page before shipping anything on top of this pack.

---

## Hacking on it

The implementation lives in two files you'll probably want to read:

- `core/gguf_split.py` — GGUF splitter + forward-hook setup
- `core/memory_tracker.py` — pre-flight VRAM projections

If you want the longer design story (component sizes, per-strategy projection math, why `cpu` is rejected for DiT), see `PLAN.md` and `MODEL_FACTS.md` in the parent directory of this repo.

PRs welcome — particularly around:

- New GGUF quant levels
- Non-T4 / asymmetric hardware profiles
- Performance numbers (per-step wall time, PCIe bottleneck measurement)

---

## Credits

Author: **The Angel Studio** ([@THE-ANGEL-AI](https://github.com/THE-ANGEL-AI))  
License: **GPL-3.0-or-later** for this code. LTX 2.3 weights themselves are under the LTX-Video Community License — see the compatibility note above.

Built on top of:

- [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — GGUF dequant
- [pollockjj/ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) — pattern we're replacing
- [dreamfast/ComfyUI-LTX2-MultiGPU](https://github.com/dreamfast/ComfyUI-LTX2-MultiGPU) — node structure reference
