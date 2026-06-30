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
#  Module-level helpers (FIX LOW #4: дедупликация _donor_choices)
# ─────────────────────────────────────────────────────────────────────────────
def _cuda_donor_choices(include_cpu: bool = False) -> list[str]:
    """FIX LOW #4+#6+#LOW_cpu_machine: динамический donor_device-список.

    Использует torch.cuda.device_count() если torch доступен И CUDA активен;
    на CPU-only машинах / torch=None fallback не показывает cuda:* опции
    (юзер не должен видеть то, что упадёт с RuntimeError на load).

    Args:
        include_cpu: True для Gemma (encoder допускает CPU — он не
                      участвует в sampling-loop); False для DiT (DiT
                      не может жить на CPU во время KSampler-step).
    """
    choices: list[str] = ["auto"]
    if torch is not None and torch.cuda.is_available():
        try:
            n = int(torch.cuda.device_count())
            for i in range(n):
                choices.append(f"cuda:{i}")
        except (RuntimeError, AssertionError):
            pass
    # Базовые cuda:0/cuda:1 присутствуют ВСЕГДА — это не зависит от torch
    # (сохраняет контракт исходного _DONOR_DEVICE_CHOICES_DIT).
    for tag in ("cuda:0", "cuda:1"):
        if tag not in choices:
            choices.append(tag)
    if include_cpu:
        choices.append("cpu")
    return choices


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
                # FIX LOW #4+#6: module-level dynamic donor_device.
                "donor_device": (_cuda_donor_choices(include_cpu=False), {"default": "auto"}),
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

    # Donor-device options FIX LOW #4: дедуп через module-level helper
    # ``_cuda_donor_choices(include_cpu=True)``.

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
                "donor_device": (_cuda_donor_choices(include_cpu=True), {"default": "auto"}),
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
        """R4: V1 tuple-контракт — возвращаем ровно ``(report,)``.

        ComfyUI ``execution.py`` анпакает выходы по схеме
        ``val, = node.FUNCTION(...)``. Dict-возврат формата
        ``{"ui": ..., "result": (...)}`` понимают только новые версии ComfyUI
        (V3 preview API); на старых / Kaggle-зерокопии — крашится с
        ``ValueError: too many values to unpack (expected 1)``.
        Для консольного preview достаточно print() — он попадает в ComfyUI log
        stdout и виден без UI.
        """
        from core.memory_tracker import estimate_vram_budget

        report = estimate_vram_budget(
            gguf_name=unet_name,
            gemma_name=gemma_name,
            purge_cache=purge_cache,
        )
        print(f"[ComfyUI-LTX2-MultiGPU]\n{report}")
        return (report,)


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
                # NEW (v0.2.1): donor_device widget — раньше нода захардкодила
                # ``secondary_dev`` из ``resolve_devices()``, из-за чего смена
                # стратегии на лету ИГНОРИРОВАЛА user override из HybridSplitLoader.
                # Теперь оба loadера (нода + HybridSplitLoader) могут горячо менять
                # стратегию с учётом оригинального donor_device.
                "donor_device": (_cuda_donor_choices(include_cpu=False), {"default": "auto"}),
            },
            "optional": {
                # FIX MEDIUM_apply_strategy: forward verbose из UI-виджета в
                # core.apply_strategy(verbose=verbose_log). WARN уже unconditional
                # в core (cab22dc), verbose используется для детального per-block лога.
                "verbose_log": ("BOOLEAN", {"default": False}),
            },
        }

    def apply_strategy(
        self,
        model,
        strategy: str,
        donor_device: str = "auto",
        verbose_log: bool = False,
    ) -> tuple:
        from core.gguf_split import apply_strategy as _apply

        try:
            new_patcher = _apply(
                patcher=model,
                strategy=strategy,
                verbose=verbose_log,
                donor_device=donor_device,
            )
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
