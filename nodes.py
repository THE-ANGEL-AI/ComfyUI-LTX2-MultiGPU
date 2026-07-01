"""Custom nodes for ComfyUI-LTX2-MultiGPU (Author: THE-ANGEL-AI).

Правила реализации (из D:\\ComfyUI\\Docs\\docs\\00-AGENT-RULES.md):
  R1 — проверять реальный API в D:\\ComfyUI\\comfy\\{model_management,model_patcher,sd}.py
  R2 — V1 API (NODE_CLASS_MAPPINGS / INPUT_TYPES) — НЕ мешать с V3
  R3 — НЕ хардкодить cuda:0/cuda:1; тянуть устройство из mm.get_torch_device()
  R4 — типы строго MODEL / CLIP / VAE / LATENT (FUNCTION возвращает tuple, длина ⇔ RETURN_TYPES)
  R5/R6 — уникальные префиксы в ключах (LTX2_MultiGPU_...), тяжёлые импорты в try/except

REFACTOR: стратегии сплита импортируются из core.gguf_split.STRATEGIES
(single source of truth).

v0.6.0-pre UI REWORK (per Agent_Info/node_fyx.md):
  - CATEGORY → nested hierarchy: ``THE-ANGEL-AI/LTX2`` (loaders + strategy) +
    ``THE-ANGEL-AI/Utilities`` (parking, diagnostics, sage).
  - DISPLAY_NAME → English with emoji per node_fyx.md: 🧠 Load, 📝 Load,
    🎨 Load, 🎯 Switch, 🅿️ Park, 💾 Diagnostics, ⚡ Sage.
  - Widget key renames (clearer for users):
      unet_name       → gguf_model
      split_strategy  → split_mode
      donor_device    → memory_gpu
      vae_name        → vae_model
      clip_name1      → clip_model
      projection_name → projection_path
      gemma_name      → clip_model (unified with GemmaHybrid)
      park_in_cpu     → park_model
      eject_models    → unload_after_generation
      verbose_log     → verbose
      strategy        → mode (in DeviceStrategy)
      purge_cache     → clear_cache_after
  - All renames are HARD RENAMES (no backward-compat aliases per user request
    to clean up dead code immediately).

⚠️ BREAKING CHANGE: existing workflow_api.json that reference old widget
   keys (unet_name, donor_device, etc.) need widget-key migration. CHANGELOG
   v0.6.0-pre lists migration path; example_workflows/* updated.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]

try:
    import folder_paths  # предоставляется ComfyUI при запуске
    _FOLDER_PATHS_OK = True
except Exception:  # noqa: BLE001
    folder_paths = None  # type: ignore[assignment]
    _FOLDER_PATHS_OK = False

# REFACTOR-1: единый список стратегий из core/gguf_split.py — single source
# of truth. Если core добавляет/удаляет стратегию, ноды синхронизируются
# автоматически.
try:
    from .core.gguf_split import STRATEGIES as _STRATEGIES  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    # Fallback: при unit-тестах где core не подгружен — держим 5 стратегий как
    # literal, чтобы ComfyUI dropdown отрисовался корректно. core.STRATEGIES —
    # канонический source.
    _STRATEGIES = (
        "blocks_50_50",
        "blocks_30_70",
        "pipeline",
        "single_cuda0",
        "single_cuda1",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────
def _memory_gpu_choices(include_cpu: bool = False) -> list[str]:
    """Динамический список device options для ComfyUI-dropdown.

    Семантика per node_fyx.md: вся VRAM placement логика unified под
    ``memory_gpu`` widget (раньше назывался ``donor_device``). Поведение:
      - ``auto`` — ВСЕГДА первая опция.
      - ``cuda:0`` / ``cuda:1`` (и cuda:N для N>1) — добавляются ТОЛЬКО если
        ``torch.cuda.is_available() == True``. На CPU-only машинах / при
        torch=None — НЕ показываем cuda:* (юзер не должен видеть опции,
        которые упадут с RuntimeError на load). Контракт задокументирован
        в docstring v0.2.x; в реализации был баг (cuda:0/cuda:1 добавлялись
        безусловно); фикс ниже.
      - ``cpu`` — только при ``include_cpu=True`` (CLIP encoder / VAE; DiT —
        never).

    Args:
        include_cpu: True для CLIP / VAE (encoder допускает CPU); False для
                      DiT (DiT не может жить на CPU во время KSampler-step).

    Note:
        Test ``tests/test_init.py::TestMemoryGpuChoices`` дублирует контракт;
        при изменении поведения — синхронизировать тест.
    """
    choices: list[str] = ["auto"]
    if torch is not None and torch.cuda.is_available():
        try:
            n = int(torch.cuda.device_count())
            for i in range(n):
                choices.append(f"cuda:{i}")
        except (RuntimeError, AssertionError):
            pass
        # Базовые cuda:0/cuda:1 присутствуют ТОЛЬКО когда CUDA активна —
        # иначе на CPU-only машинах юзер видит опции которые гарантированно
        # упадут в RuntimeError на load().
        for tag in ("cuda:0", "cuda:1"):
            if tag not in choices:
                choices.append(tag)
    if include_cpu:
        choices.append("cpu")
    return choices


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 1. 🧠 Load LTX2 GGUF Model
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_HybridSplitLoader:
    """Загружает LTX 2.3 GGUF и распределяет DiT блоки по двум GPU.

    Заменяет ``UnetLoaderGGUFDisTorch2MultiGPU`` для LTX 2.3 GGUF.
    Возвращает (R3) ModelPatcher-совместимый объект — совместим с LoRA,
    offload, sampler. Реализация сплита лежит в core/gguf_split.py.

    NB: ``unload_after_generation`` НЕ добавляется — DiT вызывается каждый
    sampling-step, offload DiT → CPU = PCIe death и sampling stall.
    """

    NODE_ID = "LTX2_MultiGPU_HybridSplitLoader"
    # English display name with brand emoji per node_fyx.md.
    # CATEGORY prefix adds the brand tag automatically in ComfyUI's Add Node
    # menu, so we don't repeat it here. NODE_ID preserved for workflow_api
    # backward-compat (не переименовывается even in breaking v0.6.0-pre).
    DISPLAY_NAME = "🧠 Load LTX2 GGUF Model"

    FUNCTION = "load"
    CATEGORY = "THE-ANGEL-AI/LTX2"
    OUTPUT_NODE = False

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        if _FOLDER_PATHS_OK:
            try:
                gguf_choices: list = folder_paths.get_filename_list("diffusion_models") or []  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                gguf_choices = []
            gguf_opts: tuple = (gguf_choices,)
        else:
            gguf_opts = ("STRING", {"default": ""})
        return {
            "required": {
                "gguf_model": gguf_opts,
                "split_mode": (
                    list(_STRATEGIES),
                    {"default": "blocks_50_50"},
                ),
                # DiT-specific: НЕТ 'cpu' — DiT не может быть на CPU во время sampling.
                "memory_gpu": (_memory_gpu_choices(include_cpu=False), {"default": "auto"}),
                # NEW (v0.6.2-pre, WhitePaper §8.6): virtual_vram_gb — reserved VRAM gap.
                # Расширяет effective cap cuda:0 для projection / auto-strategy без alloc.
                # Clamp [0, 16] GB (safety clamp [0, 8] в projection). Default 0 — no-op.
                "virtual_vram_gb": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    def load(
        self,
        gguf_model: str,
        split_mode: str,
        memory_gpu: str,
        virtual_vram_gb: int,
        verbose: bool,
    ) -> tuple:
        """Делегирует в core.gguf_split.hybrid_split_gguf (R4: tuple-возврат)."""
        if folder_paths is None:
            raise RuntimeError(
                "folder_paths недоступен — узел должен запускаться внутри ComfyUI"
            )

        from .core.gguf_split import hybrid_split_gguf

        try:
            model_patcher = hybrid_split_gguf(
                gguf_name=gguf_model,
                strategy=split_mode,
                verbose=verbose,
                donor_device=memory_gpu,
                virtual_vram_gb=float(virtual_vram_gb),
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                f"ComfyUI-LTX2-MultiGPU: {exc}"
            ) from exc

        if verbose:
            print(
                f"[ComfyUI-LTX2-MultiGPU] Loaded {gguf_model} with split_mode "
                f"{split_mode}, memory_gpu={memory_gpu}, virtual_vram_gb={virtual_vram_gb}"
            )

        return (model_patcher,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 2. 📝 Load Dual Text Encoder
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_GemmaHybridLoader:
    """Жёсткая загрузка Gemma 3 12B FP4 как Dual CLIP: encoder + projection.

    UI совместим с DualCLIPLoaderDisTorch2MultiGPU:
      clip_model              Gemma 3 safetensors (.safetensors).
      projection_path         text_projection safetensors.
      memory_gpu              куда грузить encoder (auto / cuda:0 / cuda:1 / cpu).
                             auto ⇒ mm-derived secondary (cuda:1 в dual-GPU config).
      unload_after_generation True ⇒ после load: offload_device=CPU + soft_empty_cache.
                             NB: под recursive .to() lock (BUG-6 fix) sampler не
                             сможет blanket re-load projection обратно на cuda:0
                             после unload — используйте только если уверены.
      verbose                 per-component allocation log + VRAM snapshot.
    """

    NODE_ID = "LTX2_MultiGPU_GemmaHybridLoader"
    DISPLAY_NAME = "📝 Load Dual Text Encoder"

    FUNCTION = "load"
    CATEGORY = "THE-ANGEL-AI/LTX2"
    OUTPUT_NODE = False

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        if _FOLDER_PATHS_OK:
            try:
                enc_folder = folder_paths.get_filename_list("text_encoders") or []  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                enc_folder = []
            try:
                clip_folder = folder_paths.get_filename_list("clip") or []  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                clip_folder = []
            merged: list = list(dict.fromkeys(enc_folder + clip_folder))
            clip_opts: tuple = (merged,)
            proj_opts: tuple = (merged,)
        else:
            clip_opts = ("STRING", {"default": ""})
            proj_opts = ("STRING", {"default": ""})
        return {
            "required": {
                "clip_model": clip_opts,
                "projection_path": proj_opts,
                "memory_gpu": (_memory_gpu_choices(include_cpu=True), {"default": "auto"}),
                "unload_after_generation": ("BOOLEAN", {"default": False}),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    def load(
        self,
        clip_model: str,
        projection_path: str,
        memory_gpu: str,
        unload_after_generation: bool,
        verbose: bool,
    ) -> tuple:
        from .core.gguf_split import load_gemma_hybrid

        clip = load_gemma_hybrid(
            encoder_name=clip_model,
            projection_name=projection_path,
            verbose=verbose,
            donor_device=memory_gpu,
            eject_models=unload_after_generation,
        )

        return (clip,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 3. 💾 VRAM Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_MemoryDiagnostics:
    """Pre-flight VRAM checker: dry-load прогон + nvidia-smi лог в консоль."""

    NODE_ID = "LTX2_MultiGPU_MemoryDiagnostics"
    DISPLAY_NAME = "💾 VRAM Diagnostics"

    FUNCTION = "diagnose"
    CATEGORY = "THE-ANGEL-AI/Utilities"
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        if _FOLDER_PATHS_OK:
            try:
                gguf_choices: list = folder_paths.get_filename_list("diffusion_models") or []  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                gguf_choices = []
            try:
                clip_enc = folder_paths.get_filename_list("text_encoders") or []  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                clip_enc = []
            try:
                clip_clip = folder_paths.get_filename_list("clip") or []  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                clip_clip = []
            clip_choices: list = list(dict.fromkeys(clip_enc + clip_clip))
            gguf_opts: tuple = (gguf_choices,)
            clip_opts: tuple = (clip_choices,)
        else:
            gguf_opts = ("STRING", {"default": ""})
            clip_opts = ("STRING", {"default": ""})
        return {
            "required": {
                "gguf_model": gguf_opts,
                "clip_model": clip_opts,
                # NEW (v0.6.2-pre, WhitePaper §8.6): virtual_vram_gb widget mirror
                # HybridSplitLoader / DeviceStrategy — отображается в report
                # и применяется к auto-select strategy. Default 0 = no-op.
                "virtual_vram_gb": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                # Per node_fyx.md: parameter naming follows English "clear cache
                # after" — clearer for non-native speakers than "purge".
                "clear_cache_after": ("BOOLEAN", {"default": True}),
            }
        }

    def diagnose(
        self,
        gguf_model: str,
        clip_model: str,
        virtual_vram_gb: int,
        clear_cache_after: bool,
    ) -> tuple:
        """R4: V1 tuple-контракт — возвращаем ровно ``(report,)``."""
        from .core.memory_tracker import estimate_vram_budget

        report = estimate_vram_budget(
            gguf_name=gguf_model,
            gemma_name=clip_model,
            purge_cache=clear_cache_after,
            virtual_vram_gb=float(virtual_vram_gb),
        )
        print(f"[ComfyUI-LTX2-MultiGPU]\n{report}")
        return (report,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 4. 🎯 Switch GPU Strategy
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_DeviceStrategy:
    """Hot-switch split layout без reload модели."""

    NODE_ID = "LTX2_MultiGPU_DeviceStrategy"
    DISPLAY_NAME = "🎯 Switch GPU Strategy"

    FUNCTION = "apply_strategy"
    CATEGORY = "THE-ANGEL-AI/LTX2"
    OUTPUT_NODE = False

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                # Per node_fyx.md: ``strategy`` → ``mode`` (shorter, matches
                # ComfyUI native Sampler.language convention).
                "mode": (
                    list(_STRATEGIES),
                    {"default": "blocks_50_50"},
                ),
                # NEW (v0.2.1) / v0.6.0-pre renamed: honor user override from
                # HybridSplitLoader; cpu отвергнут для DiT в INPUT_TYPES.
                "memory_gpu": (_memory_gpu_choices(include_cpu=False), {"default": "auto"}),
                # NEW (v0.6.2-pre, WhitePaper §8.6): required widget mirror HybridSplitLoader.
                # Hot-switch strategy без reload применяется — needs virtual_vram_gb
                # для consistency с HybridSplitLoader'ом (default 0 → no change).
                "virtual_vram_gb": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
            },
            "optional": {
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    def apply_strategy(
        self,
        model,
        mode: str,
        memory_gpu: str = "auto",
        virtual_vram_gb: int = 0,
        verbose: bool = False,
    ) -> tuple:
        from .core.gguf_split import apply_strategy as _apply

        try:
            new_patcher = _apply(
                patcher=model,
                strategy=mode,
                verbose=verbose,
                donor_device=memory_gpu,
                virtual_vram_gb=float(virtual_vram_gb),
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                f"ComfyUI-LTX2-MultiGPU: {exc}"
            ) from exc
        return (new_patcher,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 5. 🅿️ Park DiT (VRAM ↔ CPU)
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_VRAMParking:
    """Временно убирает DiT из VRAM для освобождения памяти под VAE/upscale.

    Между Pass 1 (KSampler) и Pass 2 (KSampler) ComfyUI делает:
      VAE decode → upscale → VAE encode.
    DiT блоки (~9 GB на каждой карте) конкурируют с VAE/upscaler за VRAM.

    Парковка переносит ВСЕ DiT блоки + служебные слои на CPU, освобождая
    VRAM под VAE-стадии. После VAE encode — распарковка возвращает блоки
    на исходные GPU в точности с исходной стратегией.
    """

    NODE_ID = "LTX2_MultiGPU_VRAMParking"
    DISPLAY_NAME = "🅿️ Park DiT (VRAM ↔ CPU)"

    FUNCTION = "apply"
    CATEGORY = "THE-ANGEL-AI/Utilities"
    OUTPUT_NODE = False

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                # Per node_fyx.md: ``park_model`` boolean toggle (was park_in_cpu).
                "park_model": ("BOOLEAN", {"default": True,
                    "label_on": "Запарковать DiT (CPU)",
                    "label_off": "Вернуть DiT (GPU)"}),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    def apply(
        self,
        model,
        park_model: bool,
        verbose: bool = False,
    ) -> tuple:
        from .core.vram_parking import park_dit, unpark_dit

        if park_model:
            park_dit(model)
        else:
            unpark_dit(model)

        if verbose:
            state = "запаркован" if park_model else "распаркован"
            print(
                f"[ComfyUI-LTX2-MultiGPU] VRAMParking: DiT {state} "
                f"(park_model={'True' if park_model else 'False'})"
            )

        return (model,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 6. ⚡ SageAttention (T4 Turbo)
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_SageAttention:
    """Ускоритель внимания через SageAttention-SM75 (оптимизация под T4/SM75).

    Заменяет стандартное scaled_dot_product_attention на квантованное
    внимание (INT8 QK^T + FP16 PV) через SageAttention-SM75 — форк,
    адаптированный для Turing SM75 (T4) через Triton fallback.

    Ожидаемое ускорение: ~1.5x end-to-end (внимание = 25-35% DiT compute).

    Автоопределение: если sageattn НЕ установлен — тихо возвращает
    модель без патча (стандартное внимание).
    """

    NODE_ID = "LTX2_MultiGPU_SageAttention"
    DISPLAY_NAME = "⚡ SageAttention (T4 Turbo)"

    FUNCTION = "apply"
    CATEGORY = "THE-ANGEL-AI/Utilities"
    OUTPUT_NODE = False

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True,
                    "label_on": "Включить SageAttn",
                    "label_off": "Выключить"}),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    def apply(
        self,
        model,
        enable: bool,
        verbose: bool = False,
    ) -> tuple:
        from .core.sage_attention import get_sageattn_patch

        # Clone patcher чтобы не мутировать оригинальный граф
        # (shallow copy model_options).
        patcher = model.clone()

        if enable:
            patch = get_sageattn_patch(verbose=verbose)
            if patch:
                opts = patcher.model_options.setdefault("attention_patch", {})
                opts.update(patch)
                if verbose:
                    print(
                        "[ComfyUI-LTX2-MultiGPU] SageAttention патч "
                        "установлен в model_options."
                    )
            elif verbose:
                print(
                    "[ComfyUI-LTX2-MultiGPU] SageAttention НЕ активирован "
                    "(модуль sageattn не найден)."
                )
        else:
            # Очищаем патч если был установлен ранее в цепочке нод.
            if "attention_patch" in patcher.model_options:
                patcher.model_options["attention_patch"].pop("default", None)
                if not patcher.model_options["attention_patch"]:
                    del patcher.model_options["attention_patch"]
            if verbose:
                print("[ComfyUI-LTX2-MultiGPU] SageAttention отключён.")

        return (patcher,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 7. 🎨 Load LTX2 VAE
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_VAELoader:
    """VAE загрузчик с выбором GPU для VAE Decode/Encode.

    На T4×2 (14.5 GB каждая) VAE decode (~3-5 GB VRAM) конкурирует
    с DiT/Gemma за память. Эта нода позволяет явно выбрать GPU для
    VAE, чтобы избежать OOM между Pass 1 и Pass 2 (VAE decode → upscale
    → VAE encode).
    """

    NODE_ID = "LTX2_MultiGPU_VAELoader"
    DISPLAY_NAME = "🎨 Load LTX2 VAE"

    FUNCTION = "load"
    CATEGORY = "THE-ANGEL-AI/LTX2"
    OUTPUT_NODE = False

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        if _FOLDER_PATHS_OK:
            try:
                vae_choices: list = folder_paths.get_filename_list("vae") or []  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                vae_choices = []
            vae_opts: tuple = (vae_choices,)
        else:
            vae_opts = ("STRING", {"default": ""})
        return {
            "required": {
                "vae_model": vae_opts,
                # VAE может жить на CPU между вызовами — include_cpu=True.
                "memory_gpu": (_memory_gpu_choices(include_cpu=True), {"default": "auto"}),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    def load(
        self,
        vae_model: str,
        memory_gpu: str,
        verbose: bool,
    ) -> tuple:
        """Загружает VAE и размещает на выбранном GPU."""
        if folder_paths is None or torch is None:
            raise RuntimeError(
                "LTX2_MultiGPU_VAELoader требует ComfyUI runtime (folder_paths, torch)"
            )

        try:
            from comfy import sd as comfy_sd  # type: ignore[import-not-found]
            from comfy import model_management as mm  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"comfy.sd / model_management недоступны: {exc}"
            ) from exc

        from .core.gguf_split import resolve_devices, resolve_donor_device

        primary_dev, secondary_dev = resolve_devices()
        target_dev = resolve_donor_device(memory_gpu, primary_dev, secondary_dev)

        vae_path = folder_paths.get_full_path("vae", vae_model)
        if not vae_path:
            raise FileNotFoundError(
                f"VAE '{vae_model}' не найден в vae/"
            )

        try:
            vae = comfy_sd.load_vae(vae_path)
        except Exception as exc:
            raise RuntimeError(
                f"comfy.sd.load_vae failed для {vae_model!r}: {exc}"
            ) from exc

        if vae is None:
            raise RuntimeError(
                f"comfy.sd.load_vae вернул None для {vae_model!r}"
            )

        if hasattr(vae, "first_stage_model"):
            try:
                vae.first_stage_model.to(target_dev, non_blocking=False)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(
                        f"[ComfyUI-LTX2-MultiGPU] WARN: VAE first_stage_model.to({target_dev}) "
                        f"failed: {exc}"
                    )
        elif hasattr(vae, "to"):
            try:
                vae.to(target_dev)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(
                        f"[ComfyUI-LTX2-MultiGPU] WARN: vae.to({target_dev}) failed: {exc}"
                    )

        if verbose:
            try:
                import os
                vae_gb = os.path.getsize(vae_path) / (1024 ** 3)
            except Exception:  # noqa: BLE001
                vae_gb = 0.0
            print(
                f"[ComfyUI-LTX2-MultiGPU] VAE loaded: {vae_model} "
                f"({vae_gb:.2f} GB) @ {target_dev} (memory_gpu={memory_gpu!r})"
            )

        return (vae,)


__all__ = [
    "LTX2_MultiGPU_HybridSplitLoader",
    "LTX2_MultiGPU_GemmaHybridLoader",
    "LTX2_MultiGPU_MemoryDiagnostics",
    "LTX2_MultiGPU_DeviceStrategy",
    "LTX2_MultiGPU_VRAMParking",
    "LTX2_MultiGPU_SageAttention",
    "LTX2_MultiGPU_VAELoader",
]
