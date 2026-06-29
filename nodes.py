"""Custom nodes for ComfyUI-LTX2-MultiGPU (Author: THE-ANGEL-AI).

Правила реализации (из D:\\ComfyUI\\Docs\\docs\\00-AGENT-RULES.md):
  R1 — проверять реальный API в D:\\ComfyUI\\comfy\\{model_management,model_patcher,sd}.py
  R2 — V1 API (NODE_CLASS_MAPPINGS / INPUT_TYPES) — НЕ мешать с V3
  R3 — НЕ хардкодить cuda:0/cuda:1; тянуть устройство из mm.get_torch_device()
  R4 — типы строго MODEL / CLIP / VAE / LATENT (FUNCTION возвращает tuple, длина ⇔ RETURN_TYPES)
  R5/R6 — уникальные префиксы в ключах (LTX2_MultiGPU_...), тяжёлые импорты в try/except
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


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 1. LTX2_MultiGPU_HybridSplitLoader
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_HybridSplitLoader:
    """Заменяет UnetLoaderGGUFDisTorch2MultiGPU для LTX 2.3 GGUF.

    Возвращает (R3) ModelPatcher-совместимый объект — совместим с LoRA,
    offload, sampler. Реализация сплита лежит в core/gguf_split.py.

    UI mirror GemmaHybrid: добавлен `donor_device` (auto/cuda:0/cuda:1).
    NB: `eject_models` НЕ добавляется — DiT вызывается каждый sampling-step,
    offload DiT → CPU = PCIe death и sampling stall. cpу как donor_device
    отвергается уже в INPUT_TYPES (выбор ограничен 3-мя опциями).
    """

    NODE_ID = "LTX2_MultiGPU_HybridSplitLoader"
    DISPLAY_NAME = "LTX-2 Hybrid Split Loader"

    FUNCTION = "load"
    CATEGORY = "LTX-2 MultiGPU"
    OUTPUT_NODE = False

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    # DiT-specific donor_device: НЕТ 'cpu' — DiT не может быть на CPU во время sampling.
    # Решение принято в design-discussion: см. commit message + docstring class.
    _DONOR_DEVICE_CHOICES_DIT: tuple[str, ...] = ("auto", "cuda:0", "cuda:1")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        unet_opts: tuple = ("STRING", {"default": ""})
        if _FOLDER_PATHS_OK:
            try:
                choices = folder_paths.get_filename_list("diffusion_models")  # type: ignore[union-attr]
                if choices:
                    unet_opts = (choices,)
            except Exception:  # noqa: BLE001
                pass
        return {
            "required": {
                "unet_name": unet_opts,
                "split_strategy": (
                    ["blocks_50_50", "blocks_30_70", "pipeline", "single_cuda0", "single_cuda1"],
                    {"default": "blocks_50_50"},
                ),
                # DiT-specific: только cuda:0/cuda:1 (cpu нельзя — sampling требует GPU).
                "donor_device": (cls._DONOR_DEVICE_CHOICES_DIT, {"default": "auto"}),
                "verbose_log": ("BOOLEAN", {"default": False}),
            }
        }

    def load(
        self,
        unet_name: str,
        split_strategy: str,
        donor_device: str,
        verbose_log: bool,
    ) -> tuple:
        """Делегирует в core.gguf_split.hybrid_split_gguf (R4: tuple-возврат)."""
        if folder_paths is None:
            raise RuntimeError(
                "folder_paths недоступен — узел должен запускаться внутри ComfyUI"
            )

        from core.gguf_split import hybrid_split_gguf

        try:
            model_patcher = hybrid_split_gguf(
                gguf_name=unet_name,
                strategy=split_strategy,
                verbose=verbose_log,
                donor_device=donor_device,
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                f"ComfyUI-LTX2-MultiGPU: {exc}"
            ) from exc

        if verbose_log:
            print(
                f"[ComfyUI-LTX2-MultiGPU] Loaded {unet_name} with strategy "
                f"{split_strategy}, donor={donor_device}"
            )

        return (model_patcher,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 2. LTX2_MultiGPU_GemmaHybridLoader
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_GemmaHybridLoader:
    """Жёсткая загрузка Gemma 3 12B FP4: encoder → donor_device, projection → cuda:0.

    UI совместим с DualCLIPLoaderDisTorch2MultiGPU (см. скриншот пользователя):
      donor_device   куда грузить encoder (auto / cuda:0 / cuda:1 / cpu).
                     auto ⇒ mm-derived secondary (cuda:1 в dual-GPU config).
      eject_models   True ⇒ после load: offload_device=CPU + soft_empty_cache.
                     NB: под Risk #7 lock (.to() no-op) sampler не сможет
                     re-load projection обратно на cuda:0 после eject —
                     используйте только если уверены.
      verbose_log    печатать per-component allocation log (encoder/proj GB,
                     VRAM free до/после).
    """

    NODE_ID = "LTX2_MultiGPU_GemmaHybridLoader"
    DISPLAY_NAME = "LTX-2 Gemma Hybrid Loader"

    FUNCTION = "load"
    CATEGORY = "LTX-2 MultiGPU"
    OUTPUT_NODE = False

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)

    # Donor-device options mirror DualCLIPLoaderDisTorch2MultiGPU. Hardcoded
    # (cuda:0/cuda:1) — INPUT_TYPES вызывается до загрузки torch-стекa,
    # rompute `torch.cuda.device_count()` здесь ненадёжен.
    _DONOR_DEVICE_CHOICES: tuple[str, ...] = ("auto", "cuda:0", "cuda:1", "cpu")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        enc_opts: tuple = ("STRING", {"default": ""})
        proj_opts: tuple = ("STRING", {"default": ""})
        if _FOLDER_PATHS_OK:
            try:
                for folder in ("text_encoders", "clip"):
                    items = folder_paths.get_filename_list(folder)  # type: ignore[union-attr]
                    if items:
                        enc_opts = (items,)
                        proj_opts = (items,)
                        break
            except Exception:  # noqa: BLE001
                pass
        return {
            "required": {
                "clip_name1": enc_opts,
                "projection_name": proj_opts,
                "donor_device": (cls._DONOR_DEVICE_CHOICES, {"default": "auto"}),
                "eject_models": ("BOOLEAN", {"default": False}),
                "verbose_log": ("BOOLEAN", {"default": False}),
            }
        }

    def load(
        self,
        clip_name1: str,
        projection_name: str,
        donor_device: str,
        eject_models: bool,
        verbose_log: bool,
    ) -> tuple:
        from core.gguf_split import load_gemma_hybrid

        # load_gemma_hybrid raises RuntimeError с понятными причинами
        # (усл.). Не мапим в RuntimeError повторно — propagate as is.
        clip = load_gemma_hybrid(
            encoder_name=clip_name1,
            projection_name=projection_name,
            verbose=verbose_log,
            donor_device=donor_device,
            eject_models=eject_models,
        )

        return (clip,)


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 3. LTX2_MultiGPU_MemoryDiagnostics
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_MemoryDiagnostics:
    """Pre-flight VRAM checker: dry-load прогон + nvidia-smi лог в консоль."""

    NODE_ID = "LTX2_MultiGPU_MemoryDiagnostics"
    DISPLAY_NAME = "LTX-2 Memory Diagnostics"

    FUNCTION = "diagnose"
    CATEGORY = "LTX-2 MultiGPU"
    OUTPUT_NODE = True  # для ui={'report': [report]}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "unet_name": ("STRING", {"default": ""}),
                "gemma_name": ("STRING", {"default": ""}),
                "purge_cache": ("BOOLEAN", {"default": True}),
            }
        }

    def diagnose(
        self,
        unet_name: str,
        gemma_name: str,
        purge_cache: bool,
    ) -> tuple:
        """R4: V1 dict-контракт для нод с ui preview.

        Возвращает ``{"ui": {...}, "result": (...)}``, где длина result
        строго равна ``len(RETURN_TYPES)``. ComfyUI ``execution.py``
        использует ``result`` и ``ui`` как два раздельных поля —
        возврат 2-tuple ``(report, {"ui": ...})`` ломал анпак выходов
        по схеме R4 (``ValueError: too many values to unpack``).
        """
        from core.memory_tracker import estimate_vram_budget

        report = estimate_vram_budget(
            gguf_name=unet_name,
            gemma_name=gemma_name,
            purge_cache=purge_cache,
        )
        print(f"[ComfyUI-LTX2-MultiGPU]\n{report}")
        return {"ui": {"report": [report]}, "result": (report,)}


# ─────────────────────────────────────────────────────────────────────────────
#  Узел 4. LTX2_MultiGPU_DeviceStrategy
# ─────────────────────────────────────────────────────────────────────────────
class LTX2_MultiGPU_DeviceStrategy:
    """Переключатель глобальной стратегии распределения через mm."""

    NODE_ID = "LTX2_MultiGPU_DeviceStrategy"
    DISPLAY_NAME = "LTX-2 Device Strategy Switch"

    FUNCTION = "apply_strategy"
    CATEGORY = "LTX-2 MultiGPU"
    OUTPUT_NODE = False

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "strategy": (
                    ["blocks_50_50", "blocks_30_70", "pipeline", "single_cuda0", "single_cuda1"],
                    {"default": "blocks_50_50"},
                ),
            }
        }

    def apply_strategy(self, model, strategy: str) -> tuple:
        from core.gguf_split import apply_strategy as _apply

        try:
            new_patcher = _apply(patcher=model, strategy=strategy)
        except NotImplementedError as exc:
            raise RuntimeError(
                f"ComfyUI-LTX2-MultiGPU: {exc}"
            ) from exc
        return (new_patcher,)


__all__ = [
    "LTX2_MultiGPU_HybridSplitLoader",
    "LTX2_MultiGPU_GemmaHybridLoader",
    "LTX2_MultiGPU_MemoryDiagnostics",
    "LTX2_MultiGPU_DeviceStrategy",
]
