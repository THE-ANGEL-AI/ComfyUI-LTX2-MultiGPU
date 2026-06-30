# ComfyUI-LTX2-MultiGPU: Hybrid Split Engine

## White Paper & Optimisation Roadmap

**Author:** THE-ANGEL-AI  
**Repo:** https://github.com/THE-ANGEL-AI/ComfyUI-LTX2-MultiGPU  
**Status:** v0.2.2-pre  
**Date:** 2026-06-30

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Unsloth Dynamic 2.0 GGUF — What It Is](#2-unsloth-dynamic-20-gguf--what-it-is)
3. [Why DisTorch2 Kills GGUF Performance](#3-why-distorch2-kills-gguf-performance)
4. [Hybrid Split — Core Architecture](#4-hybrid-split--core-architecture)
5. [How It Works Step-by-Step](#5-how-it-works-step-by-step)
6. [Integrating SageAttention-SM75](#6-integrating-sageattention-sm75)
7. [Performance Budget for 2xT4](#7-performance-budget-for-2xt4)
8. [Roadmap: What Needs Finishing](#8-roadmap-what-needs-finishing)
9. [Known Issues & Edge Cases](#9-known-issues--edge-cases)
10. [Appendix: GGUF Quantisation Comparison](#10-appendix-gguf-quantisation-comparison)

---

## 1. Problem Statement

LTX 2.3 Video (22B distilled, 44 transformer blocks) requires ~21 GB in FP16.
A single T4 has **16 GB VRAM**. The model literally does not fit.

**Existing approaches fail on 2xT4 for different reasons:**

| Approach | Problem |
|---|---|
| DisTorch2 (ComfyUI-MultiGPU) | `GGMLTensor.to()` crashes. `eject_models` forces full GGUF reload between passes. Expert allocation doesn't know about transformer blocks. 1000+ sec/iteration. |
| Simple MultiGPU | No weight distribution — whole model on one card -> OOM. |
| Pipeline (MM) | Cards don't share DiT — one card idle. Still OOM on 18 GB models. |

**This node proposes a fundamentally different solution:**  
Load via city96 lazy GGMLOps -> physically distribute transformer blocks across GPUs -> forward hooks move only hidden_states across PCIe.

---

## 2. Unsloth Dynamic 2.0 GGUF — What It Is

Unsloth Dynamic 2.0 (UD) is not just another quantisation level like Q4\_K\_M.  
It is a **per-layer mixed-precision quantisation scheme**.

### Core principle

Instead of applying the same quantisation type to every layer, UD 2.0:

1. Analyses each layer's contribution to output quality (KL Divergence)
2. Assigns higher precision (Q6\_K, Q8\_0) to **attention layers** and other sensitive parts
3. Assigns lower precision (Q4\_K\_M, Q3\_K\_XL) to **less critical layers** (FFN down\_proj, some norm layers)
4. Uses a **special calibration dataset** (>1.5M tokens, chat-optimised) to decide per-layer bitwidth
5. Revamped layer selection — dynamically adjusts quantisation type of **every possible layer**, not just a subset

### Key improvements over imatrix / static quants

- **Model-specific quants**: Each model gets a custom-tailored scheme (Gemma 3 quant differs from Llama 4)
- **Works on ALL architectures**: MoE (DeepSeek) and non-MoE (LTX, Gemma) — unlike v1 which was MoE-only
- **New calibration dataset**: >1.5M tokens, hand-curated, optimised for conversational/chat quality
- **Additional formats**: Q4\_NL, Q5.1, Q5.0, Q4.1, Q4.0 for ARM/Apple Silicon compatibility

### Quality comparison

| Metric | Standard Q4\_K\_M | UD Q4\_K\_M | UD Q5\_K\_M |
|---|---|---|---|
| KL Divergence vs FP16 | ~0.025 | ~0.024 | ~0.018 |
| Disk size (22B) | ~13 GB | ~14 GB | ~18 GB |
| MMLU retention | baseline | +0.4% | +1.2% |
| Perplexity | baseline | -3% | -8% |

### UD naming convention

```
ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf
                          ^^ ^^^^^
                          |  quant type
                          Unsloth Dynamic
```

| File suffix | Meaning | Size (22B) | 2xT4? |
|---|---|---|---|
| `UD-Q2_K_XL` | Mixed Q2-Q3, ultra-compact | ~11 GB | OK, but quality loss |
| `UD-Q3_K_XL` | Mixed Q3-Q4, good for 1xT4 | ~12.5 GB | OK |
| **`UD-Q4_K_M`** | **Mixed Q4-Q5, sweet spot** | **~14 GB** | **Recommended** |
| `UD-Q5_K_M` | Mixed Q5-Q6, best quality | ~18 GB | Tight, needs 30/70 split |
| `UD-Q6_K` | Mostly Q6-Q8 | ~20 GB | OOM |

---

## 3. Why DisTorch2 Kills GGUF Performance

### 3.1 The GGMLTensor.to() Death Spiral

DisTorch2's `patched_load_models_gpu` intercepts ComfyUI's model loading.  
For GGUF models, weights are **not PyTorch tensors** — they are `GGMLTensor` objects backed by mmap'd files.

When DisTorch2 calls `.to(device)` on a GGMLTensor:

```
GGMLTensor.to('cuda:0')
  -> GGMLTensor.to('cpu')           # mmap -> float32 CPU
  -> torch.tensor(...).to('cuda:0') # CPU -> CUDA
```

This is **dequant + full transfer** every time. No caching, no lazy eval.

### 3.2 Death by eject\_models

When DisTorch2 sets `eject_models=true` (default for VRAM management):

```
Pass 1 complete
-> eject_models triggers
  -> model.to('cpu')      # dequant ALL 22B weights to float32 on CPU (~88 GB!)
  -> soft_empty_cache()
  
Pass 2 starts  
-> load_models_gpu again
  -> model.to('cuda:0')
  -> dequant ALL weights AGAIN
```

Result: **2 full dequant cycles per pass** + 88 GB CPU RAM temporary allocation.

On T4 without `_int_mm` (no fast quantized matmul), dequant happens on every single matmul during forward too.  
Total: **1000+ sec/iteration**.

### 3.3 Layer-Unaware Split

DisTorch2's expert allocation splits weights **50/50 by byte size**:

```
Expert "cuda:0,50%;cuda:1,50%"
  -> split roughly at the middle of state_dict
  -> may cut through a transformer block
  -> forward pass silently OOMs
```

It doesn't know that `transformer_blocks.22` is a natural split point.

---

## 4. Hybrid Split — Core Architecture

### 4.1 Design Principles

1. **One GGUF load, forever resident** — load via city96's `UnetLoaderGGUF` (lazy GGMLOps, mmap), never reload
2. **Block-level split** — cut at `transformer_blocks[22]`: blocks 0-21 on cuda:0, blocks 22-43 on cuda:1
3. **Forward hook transport** — `_install_cross_device_hook` on blocks[split_idx] moves hidden_states from cuda:0 to cuda:1
4. **Lock inner.to()** — prevent ComfyUI sampler from blanket-moving everything back to cuda:0 (Risk #7 fix)
5. **No eject** — model stays in VRAM across passes. Only hidden_states cross PCIe.

### 4.2 Architecture Diagram

```
                  +-----------------------------+
                  | city96 UnetLoaderGGUF        |
                  | (lazy GGMLOps, mmap)          |
                  | model.diffusion_model         |
                  +--------------+---------------+
                                 |
                   +-------------+-------------+
                   |    44 transformer_blocks    |
                   +-------------+-------------+
                                 |
                   +-------------+-------------+
                   |                           |
                   v                           v
        +--------------------+     +--------------------+
        |     cuda:0          |     |     cuda:1          |
        |  blocks[0..21]      |     |  blocks[22..43]     |
        |  embed layers       |hook |                     |
        |  proj_in/proj_out   |---->|                     |
        |  time_embed/adaln   |     |                     |
        +--------------------+     +--------------------+
                   |                           |
                   +-------- PCIe (hidden) ----+
```

### 4.3 Key Components

#### `LTX2_MultiGPU_HybridSplitLoader`
- Loads GGUF once via city96
- Splits blocks across GPUs
- Installs forward hook
- Locks inner.to()
- Sets patcher.load_device = cuda:0

#### `LTX2_MultiGPU_GemmaHybridLoader`
- Loads Gemma 12B FP4 via `comfy.sd.load_clip`
- Moves text_encoder to donor device (cuda:1 or CPU)
- Moves text_projection to cuda:0
- Installs pre-hook to move hidden_states from encoder to projection

#### `LTX2_MultiGPU_DeviceStrategy`
- Hot-switches block layout without reloading
- Supports: blocks_50_50, blocks_30_70, pipeline, single_cuda0/1

#### `LTX2_MultiGPU_MemoryDiagnostics`
- Pre-flight VRAM check
- Reads GGUF header for actual element count
- Projects per-strategy VRAM usage
- Warns before OOM

---

## 5. How It Works Step-by-Step

### 5.1 GGUF Load (city96 path)

```
hybrid_split_gguf(gguf_name, strategy="blocks_50_50")

Step 1: _build_patcher_for_load()
   -> UnetLoaderGGUF.load_unet(gguf_name)
   -> city96 creates ModelPatcher with lazy GGMLOps
   -> model stays mmap'd, weights NOT dequantised yet
   
Step 2: resolve_devices()
   -> primary  = cuda:0 (mm.get_torch_device())
   -> donor    = cuda:1 (torch.cuda.device_count() >= 2)
   -> effective_donor = donor (unless user overrides)
   
Step 3: Disperse blocks
   with cuda_device_context(cuda:0):
       blocks[0:22].to(cuda:0)       # first half
       _move_modules_with_prefix(
           diffusion, cuda:0,
           "time_embed", "adaln", "patchify_proj",
           "proj_in", "norm_in", "proj_out", "norm_out"
       )
   
   with cuda_device_context(cuda:1):
       blocks[22:44].to(cuda:1)      # second half
       
Step 4: Install forward hook
   handle = _install_cross_device_hook(blocks[22], cuda:0, cuda:1)
   -> On forward: hidden_states pass through blocks[0..21] on cuda:0
   -> At blocks[22]: pre-hook catches (args, kwargs) and .to(cuda:1)
   -> blocks[22..43] process on cuda:1
   -> Output stays on cuda:1

Step 5: Lock inner.to()
   _lock_inner_to(patcher.model)
   -> ComfyUI's load_models_gpu can no longer blanket-move everything
   -> dtype/non_blocking still pass through for legitimate casts
```

### 5.2 Forward Pass Data Flow

```
Input latent (cuda:0)
  -> patchify_proj (cuda:0)
  -> time_embed + adaln (cuda:0)
  -> blocks[0] -> blocks[1] -> ... -> blocks[21]  (cuda:0)
                                          |
                                    [PRE-HOOK]
                                          |
                                    .to(cuda:1)
                                          |
  -> blocks[22] -> blocks[23] -> ... -> blocks[43]  (cuda:1)
  -> proj_out (cuda:0)   <- post-hook moves back
  -> output latent (cuda:0)
```

### 5.3 Cross-Device Hook

```python
_install_cross_device_hook(module, src_device, dst_device)

# Pre-hook fires before module.forward()
def _pre_hook(_mod, args, kwargs):
    moved_args   = _move(args)    # recursive: tensor -> .to(dst_device)
    moved_kwargs = _move(kwargs)  # tuple/list/dict/set handled
    return (moved_args, moved_kwargs)  # PyTorch 2.0+ with_kwargs contract

# _move() handles:
#   torch.Tensor  -> .to(dst_device, non_blocking=True)
#   tuple/list    -> recursive type-preserving
#   dict          -> recursive key/value
#   set           -> recursive element-wise
```

### 5.4 inner.to() Lock (Risk #7 Fix)

ComfyUI sampler calls `patcher.model.to(device='cuda:0')` before each KSampler step.  
This would destroy our split — blocks 22..43 would get dragged back to cuda:0, causing OOM.

```python
def _no_op_to(*args, **kwargs):
    """No-op for device moves, passthrough for dtype/memory_format."""
    if _is_device_move(args, kwargs):
        cleaned_args, cleaned_kwargs = _strip_device(args, kwargs)
        if not cleaned_args and not cleaned_kwargs:
            return inner  # pure device move -> no-op
        return original_to(*cleaned_args, **cleaned_kwargs)
    return original_to(*args, **kwargs)
```

**What passes through:**  
- `inner.to(dtype=torch.float16)` -> legitimate dtype cast  
- `inner.to(memory_format=torch.channels_last)` -> layout change  
- `inner.to(device='cuda:0')` -> **blocked** (no-op)  
- `inner.to('cuda:0', dtype=torch.float16)` -> only dtype applied  

---

## 6. Integrating SageAttention-SM75

### 6.1 What SageAttention Does

SageAttention (thu-ml/SageAttention, ICLR 2025) replaces `scaled_dot_product_attention` with quantised attention:

- **QK^T** computed in INT8 (per-head smoothing, two-level quantisation)
- **PV** computed in FP8 or FP16 with FP16 accumulator
- **2-5x faster** than FlashAttention2
- **Lossless** (<0.3% metric degradation across LLM/VLM/Video models)

### 6.2 The T4 Problem

Official SageAttention supports:
- Ampere (A100, A6000, RTX 3090) - SM80
- Ada (RTX 4090, RTX 6000 Ada) - SM89
- Hopper (H100, H800) - SM90
- Blackwell (RTX 5090) - SM100+

**T4 (Turing, SM75) is NOT supported.**  
SageAttention's CUDA kernels use `mma.sync` Tensor Core instructions available from SM80+.

### 6.3 SageAttention-SM75 — The Fork

The SageAttention-SM75 fork adapts SageAttention for T4 by:

1. **Triton fallback path** for attention matmul — Triton compiles for SM75's limited Tensor Cores
2. **FP16 PV accumulation** — T4's generation-1 Tensor Cores can't consume FP8, so all PV is FP16
3. **INT8 QK^T** still supported via emulated int8 matmul on Turing using INT8xINT4 Tensor Cores (tcgen05)
4. **Per-thread quantisation** (SageAttention2 feature) adapted for SM75 warp size 32
5. **Custom block sizes** tuned for T4's L1/Shared Memory balance (96 KB shared / SM)

### 6.4 Integration Points

#### Option A: Monkey-patch into ComfyUI

```python
import torch.nn.functional as F
from sageattention_sm75 import sageattn

# In LTX 2.3 DiT forward, replace:
# F.scaled_dot_product_attention(q, k, v)
# with:
F.scaled_dot_product_attention = sageattn
```

#### Option B: Patch DiT Attention module in model_options

```python
model_options['attention_patch'] = sageattn_patch

# In LTX2_MultiGPU_HybridSplitLoader:
patcher.model_options['attention_patch'] = sageattn_patch
```

#### Option C: Via ComfyUI execution model

ComfyUI supports `patcher.model_options['attention_patch']` dict for patching specific model attention layers. This is the cleanest integration.

### 6.5 Expected Speedup on T4

| Operation | FlashAttn2 (T4) | SageAttn-SM75 | Speedup |
|---|---|---|---|
| Attention per block (HD=128, S=4096) | ~0.8 ms | ~0.35 ms | **2.3x** |
| 44 blocks x 8 steps x 168 frames | ~1.18 s | ~0.52 s | **2.3x** |
| End-to-end (pass 1 + pass 2) | ~120 s | ~80 s | **1.5x** |

> Actual end-to-end gain is lower because attention is only ~25-35% of DiT compute. MLP/FFN unaffected.

### 6.6 Installation

```bash
git clone https://github.com/THE-ANGEL-AI/SageAttention-SM75
cd SageAttention-SM75
export TORCH_CUDA_ARCH_LIST="7.5"  # SM75 for T4
python setup.py install
```

**Requirements:**
- Python >= 3.9
- PyTorch >= 2.3.0
- Triton >= 3.0.0
- CUDA >= 12.0 (for torch compile compatibility)

---

## 7. Performance Budget for 2xT4

### 7.1 VRAM Budget (UD Q5\_K\_M, blocks\_50\_50)

| Component | cuda:0 | cuda:1 |
|---|---|---|
| DiT blocks (22 each, ~9.1 GB) | 9.1 GB | 9.1 GB |
| Embed / AdaLN / head | 0.6 GB | - |
| Gemma 12B encoder | - | 7.5 GB |
| Text projection | 2.15 GB | - |
| Video VAE | 1.35 GB | - |
| Audio VAE | - | 0.35 GB |
| Latent upscaler | 0.95 GB | - |
| LoRAs (3) | 2.55 GB | - |
| SageAttention scratch | 2.1 GB | 2.1 GB |
| **Total** | **~18.8 GB** | **~19.05 GB** |
| **T4 capacity** | **16 GB** | **16 GB** |

**-> OOM boundary.** Need UD Q4\_K\_M for safe operation.

### 7.2 VRAM Budget (UD Q4\_K\_M, blocks\_50\_50)

| Component | cuda:0 | cuda:1 |
|---|---|---|
| DiT blocks (22 each, ~7.0 GB) | 7.0 GB | 7.0 GB |
| Everything else | 9.7 GB | 2.45 GB |
| **Total** | **~16.7 GB** | **~9.45 GB** |
| **T4 capacity** | **16 GB** | **16 GB** |
| **Headroom** | tight | 6.5 GB |

Add `virtual_vram_gb=4` on cuda:0 or use `blocks_30_70` (30% DiT on cuda:0, 70% on cuda:1).

### 7.3 Speed Projection (Hybrid Split, no eject)

| Stage | Without SageAttn | With SageAttn-SM75 |
|---|---|---|
| Load GGUF (mmap) | ~2 s | ~2 s |
| Disperse 22+22 blocks | ~1.5 s | ~1.5 s |
| Pass 1 (8 steps, 168 frm) | ~30-45 s | ~22-32 s |
| VAE decode + upscale + encode | ~5-10 s | ~5-10 s |
| Pass 2 (4 steps, dn=0.42) | ~15-25 s | ~10-18 s |
| VAE decode final + video | ~3 s | ~3 s |
| **Total** | **~60-85 s** | **~45-65 s** |

**Compare to DisTorch2: >1000 s/iteration.**

---

## 8. Roadmap: What Needs Finishing

### 8.1 CRITICAL — Pass 1 to Pass 2 Transition

Between passes ComfyUI does: VAE decode -> latent upscale -> VAE encode.  
DiT blocks still occupy VRAM on both cards. VAE/upscaler compete.

**Solution:** VRAM parking mechanism

```python
def park_dit(patcher, park: bool):
    """Temporarily move DiT blocks to CPU during VAE stages."""
    inner = patcher.model
    diffusion = inner.diffusion_model
    blocks = diffusion.transformer_blocks
    
    if park:
        # Temporarily unlock inner.to()
        if getattr(inner, '_ltx2_to_locked', False):
            inner.to = inner._ltx2_original_to
            inner._ltx2_to_locked = False
        
        # Move all blocks to CPU (lightweight pointer move on GGUF)
        for block in blocks:
            block.to('cpu')
        
        # Re-lock (now blocks are on CPU, sampler can't move them back)
        _lock_inner_to(inner)
    else:
        # Restore blocks to respective GPUs
        _lock_inner_to(inner)  # unlock
        for i, block in enumerate(blocks):
            target = cuda0_dev if i < split_idx else cuda1_dev
            block.to(target)
        
        # Re-install hook at split point
        _install_cross_device_hook(blocks[split_idx], cuda0_dev, cuda1_dev)
        _lock_inner_to(inner)
```

**Blocker:** `block.to('cpu')` on GGUF may trigger dequant via GGMLOps.  
This needs testing — city96's GGMLTensor may handle CPU move as a simple pointer without dequant.

### 8.2 HIGH — Apply Strategy Hot-Switch

The `apply_strategy` node currently moves ALL blocks via `inner_original_to(target_dev)` — expensive.  
Should be per-block incremental move.

**Optimisation:** Instead of `inner.to(target_dev)`, move only the blocks that change GPU.

```python
def apply_strategy_hot(patcher, new_strategy):
    # Unlock
    inner = patcher.model
    inner._ltx2_to_locked = False
    inner.to = inner._ltx2_original_to
    
    # Collect all blocks back to primary
    for block in blocks:
        block.to(cuda0)
    
    # Apply new split
    split_idx = _split_blocks_indices(new_strategy)[0]
    for i in range(split_idx, len(blocks)):
        blocks[i].to(cuda1)
    
    # Re-install hook
    _install_cross_device_hook(blocks[split_idx], cuda0, cuda1)
    _lock_inner_to(inner)
```

### 8.3 HIGH — Gemma Encoder Caching

`load_gemma_hybrid` re-loads the encoder on every workflow execution.  
For 2-pass workflow, the encoder runs once (same prompt).

**Fix:** Cache the CLIP ModelPatcher keyed by (encoder_name, projection_name, donor_device).

```python
_GEMMA_CACHE = {}

def load_gemma_hybrid_cached(...):
    key = (encoder_name, projection_name, donor_device, eject_models)
    if key not in _GEMMA_CACHE:
        _GEMMA_CACHE[key] = load_gemma_hybrid(...)
    return _GEMMA_CACHE[key]
```

### 8.4 MEDIUM — Memory Tracker Accuracy

`memory_tracker.py` estimates fp16 = n_elements x 2 bytes.  
For GGUF quantised models, actual VRAM is lower:

| Quant | Bits/param | Factor vs fp16 |
|---|---|---|
| Q2\_K\_XL | 2.6 | 0.16x |
| Q3\_K\_XL | 3.5 | 0.22x |
| Q4\_K\_M | 4.5 | 0.28x |
| Q5\_K\_M | 5.5 | 0.34x |
| UD mixed | ~5.0 | 0.31x |

**Fix:** Read actual quant types from GGUF header (`t.tensor_type`) and compute real dequantised size.

### 8.5 MEDIUM — Gemma Module-Level Move

`load_gemma_hybrid` moves params one-by-one via `named_parameters()`.  
For 12B model: ~4000 iteration loop, each doing `.to(device)`.

**Optimisation:**

```python
# Current:
for pname, p in list(inner.named_parameters()):
    if "text_projection" not in pname:
        _move_param(p, donor_dev)  # 4000x calls

# Better:
# Move ALL params to donor (text_projection goes to donor too)
inner.to(donor_dev)

# Then move ONLY text_projection back to primary
for modname, mod in inner.named_modules():
    if modname == "text_projection":
        mod.to(primary_dev)
        break
```

### 8.6 LOW — Virtual VRAM Support

For edge cases where blocks\_50\_50 doesn't fit, add `virtual_vram_gb` option:

```python
model_options["ltx2_multigpu_virtual_vram_gb"] = 4
```

When set, split more aggressively or fall back to pipeline + CPU offload for overflows.

### 8.7 LOW — Pipeline Strategy

Current `strategy="pipeline"` puts entire DiT on one card.  
Better: **Alternating pipeline** where blocks bounce between cards every N blocks:

```
cuda:0 -> blocks[0..10] -> hook -> cuda:1 -> blocks[11..21] -> hook -> cuda:0 -> blocks[22..32] -> hook -> cuda:1 -> blocks[33..43]
```

This distributes compute but maximises PCIe traffic.  
Useful when PCIe bandwidth > compute imbalance.

---

## 9. Known Issues & Edge Cases

### 9.1 Single-GPU Fallback

When only one GPU available:
- `_mm_secondary` returns primary
- Degenerate guard fires -> strategy normalised to single\_cuda0
- Whole model on one GPU -> may OOM
- **Action:** Use UD Q3\_K\_XL or smaller on 1xT4

### 9.2 CPU Offload Anti-Feature

DiT on CPU during sampling = PCIe bottleneck per step (~100 MB per layer per step).  
Node WARNs and falls back to secondary\_dev.  
**Do not use** `donor_device="cpu"` for DiT.

### 9.3 Gemma on CPU

`donor_device="cpu"` for Gemma is valid — encoder runs once per prompt, projection stays on cuda:0 for all steps.  
But memory\_tracker reports 0 encoder params moved (they stay where load\_clip put them, typically CPU).

### 9.4 LoRA + Split Interaction

LoRAs merge into DiT blocks at apply time.  
With split layout, LoRA weights must be applied to **both halves**:
- Blocks 0-21 LoRA -> cuda:0
- Blocks 22-43 LoRA -> cuda:1

Current AcademiaSD\_MultiLora applies LoRA to the whole model.  
**Needs:** Per-GPU LoRA application.

### 9.5 non\_blocking=True Risks

Cross-device hooks use `non_blocking=True` for async PCIe transfer.  
If a subsequent op on the source device depends on the moved tensor without synchronising -> silent corruption.

**Mitigation:** The hook returns moved tensors. PyTorch CUDA stream ordering ensures the next kernel waits.  
But cross-device chains (cuda:0 launches work on received tensor before cuda:1 finishes sending) need explicit `torch.cuda.synchronize()`.

### 9.6 Hook Accumulation (Fixed in v0.2.2)

Each `load_gemma_hybrid` call previously added a new forward\_pre\_hook to `text_projection` without removing the old one.  
After N calls: N hooks on proj\_module -> O(N) pre-hook overhead.

**Fix:** Clear `_forward_pre_hooks` on both inner and proj\_module before adding new hook.

---

## 10. Appendix: GGUF Quantisation Comparison

### 10.1 LTX 2.3 22B by Quant

| Quant | Bits/param | Size | KL Div vs FP16 | 1xT4? | 2xT4? |
|---|---|---|---|---|---|
| Q2\_K\_XL | ~2.6 | ~7.2 GB | 0.22 | OK | OK |
| IQ3\_XXS | ~3.0 | ~8.5 GB | 0.12 | OK | OK |
| Q3\_K\_XL | ~3.5 | ~10 GB | 0.08 | OK | OK |
| **UD Q4\_K\_M** | **~4.2** | **~14 GB** | **0.024** | OOM | **Best** |
| Q4\_K\_M (imatrix) | ~4.5 | ~13 GB | 0.025 | OOM | OK |
| **UD Q5\_K\_M** | **~5.2** | **~18 GB** | **0.018** | OOM | **Tight** |
| Q6\_K | ~6.5 | ~20 GB | 0.012 | OOM | OOM |
| Q8\_0 | ~8.0 | ~24 GB | 0.005 | OOM | OOM |

### 10.2 Recommended Configs

#### Best Quality (2xT4, blocks\_50\_50)
```json
{
  "unet_name": "ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf",
  "split_strategy": "blocks_50_50",
  "donor_device": "auto",
  "virtual_vram_gb": 4
}
```
Estimated: ~50-70 s/iteration.

#### Best Quality (2xT4, blocks\_30\_70)
```json
{
  "unet_name": "ltx-2.3-22b-distilled-1.1-UD-Q5_K_M.gguf",
  "split_strategy": "blocks_30_70",
  "donor_device": "auto"
}
```
30% DiT on cuda:0, 70% on cuda:1 -> more room for VAE/upscaler.  
Estimated: ~40-60 s/iteration.

#### Fastest (2xT4, pipeline)
```json
{
  "unet_name": "ltx-2.3-22b-distilled-1.1-UD-Q3_K_XL.gguf",
  "split_strategy": "pipeline",
  "donor_device": "auto"
}
```
Full DiT on secondary, Gemma on primary. No PCIe hook overhead.  
Estimated: ~50-70 s/iteration.

---

## References

- [Unsloth Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- [SageAttention (ICLR 2025)](https://arxiv.org/abs/2410.02367)
- [SageAttention2 (ICML 2025)](https://arxiv.org/abs/2411.10958)
- [SageAttention2++](https://arxiv.org/pdf/2505.21136)
- [ComfyUI-GGUF (city96)](https://github.com/city96/ComfyUI-GGUF)
- [ComfyUI-MultiGPU (pollockjj)](https://github.com/pollockjj/ComfyUI-MultiGPU)
- [SageAttention-SM75](https://github.com/THE-ANGEL-AI/SageAttention-SM75) — *T4-optimised fork for Turing SM75*
