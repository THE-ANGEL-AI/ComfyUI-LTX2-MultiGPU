"""Regression tests for core/memory_tracker.py + core/gguf_split.py helpers.

Pure-stdlib unittest (no pytest required). Run via:
    python -m unittest tests.test_memory_tracker -v
    # or directly:
    python tests/test_memory_tracker.py

Target coverage:
  1. ``_project(strategy, dit_gb, gemma_gb, components)`` — per-strategy
     projection math (after v0.2.1 fix: loras_estimate is now in pipeline-projection).
     NB (v0.2.2-pre+): ``_project`` — module-level. Tests work against the real
     production function, NOT a MIRROR-copy of math (это устраняет silent-
     divergence между in-package и standalone-CLI ``scripts/diagnostic.py``).
  2. ``COMPONENT_FOOTPRINT_GB`` keys — синхронизированы с diagnostic.py v0.2.1.
  3. ``gguf_split.classify_tensor`` — DiT tensor-name → category mapping.
  4. ``gguf_split._resolve_donor_device`` — donor_device → torch.device mapping
     (auto / cuda:0 / cuda:1 / cpu fallback).
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBMODULE_ROOT = HERE.parent
if str(SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMODULE_ROOT))

from core import gguf_split  # noqa: E402
from core.gguf_split import (  # noqa: E402
    DIFFUSION_BLOCK_PREFIX,
    DIFFUSION_EMBED_PREFIXES,
    _resolve_donor_device,
    classify_tensor,
)
from core import memory_tracker  # noqa: E402
from core.memory_tracker import COMPONENT_FOOTPRINT_GB, _project  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Block 1: regression guards для v0.2.1 polish
# ════════════════════════════════════════════════════════════════════════════
class TestV021PolishGuards(unittest.TestCase):
    """Проверки фиксов из v0.2.1 CHANGELOG:

    - ``sage_attention_scratch`` key — должна быть в COMPONENT_FOOTPRINT_GB
      (HIGH-1: была key-mismatch с diagnostic.py, где был ``sage_scratch``).
    - ``_project("pipeline")`` — должна включать loras_estimate
      (MED-4: раньше silent-bug projection для pipeline).
    """

    def test_sage_attention_scratch_key_present(self):
        """HIGH-1 fix: ключ ``sage_attention_scratch`` синхронизирован."""
        self.assertIn(
            "sage_attention_scratch",
            COMPONENT_FOOTPRINT_GB,
            "v0.2.1 polish: ключ должен быть sage_attention_scratch.",
        )
        # Значение не должно быть 0 (footprint имеет смысл).
        self.assertEqual(
            COMPONENT_FOOTPRINT_GB["sage_attention_scratch"],
            2.1,
            "sage_attention_scratch должен быть 2.1 GB (~sage-attn peak).",
        )

    def test_old_sage_scratch_key_removed(self):
        """HIGH-1 follow-up: после sync diagnostic.py тоже переименован.

        Этот тест — defensive guard против повторной рассинхронизации:
        старый ключ ``sage_scratch`` не должен возвращать данные сам по себе
        (он оставлен только как fallback в diagnostic.py _compute_other)."""
        # В memory_tracker.py — только canonical key.
        self.assertNotIn(
            "sage_scratch",
            COMPONENT_FOOTPRINT_GB,
            "v0.2.1: старый ключ должен быть удалён из memory_tracker.",
        )


# ════════════════════════════════════════════════════════════════════════════
# Block 2: core/memory_tracker.py._project() — direct calls to real function
# ════════════════════════════════════════════════════════════════════════════
class TestProjectPerStrategy(unittest.TestCase):
    """Проверка расчёта VRAM-загрузки для каждой strategy после v0.2.1.

    NB (v0.2.2-pre+): ``memory_tracker._project`` теперь module-level (был
    nested closure в ``estimate_vram_budget``) — tests работают с реальной
    production-функцией, не MIRROR-копией math. Это устраняет silent-divergence
    между in-package и standalone-CLI ``scripts/diagnostic.py`` проекциями.
    """

    def test_pipeline_includes_loras_on_cuda1(self):
        """MED-4: pipeline-projection теперь учитывает LoRы на cuda:1.

        До v0.2.1 cuda:1 был ``dit_gb + other_cuda1`` — silent-bug
        under-reporting на 2.55 GB. Сейчас dit_gb + other_cuda1 + loras.
        """
        c0, c1 = _project(
            "pipeline", dit_gb=10.0, gemma_gb=7.5, components=COMPONENT_FOOTPRINT_GB,
        )
        c1_loras_expected = (
            10.0
            + COMPONENT_FOOTPRINT_GB["audio_vae"]
            + COMPONENT_FOOTPRINT_GB["sage_attention_scratch"]
            + COMPONENT_FOOTPRINT_GB["loras_estimate"]
        )
        self.assertAlmostEqual(c1, c1_loras_expected, places=4)
        self.assertGreater(
            c1,
            COMPONENT_FOOTPRINT_GB["gemma_fp4"] * 0.5,
            "pipeline cuda:1 теперь >= 10 ГБ (LoRы включены).",
        )

    def test_blocks_50_50_loras_only_on_cuda0(self):
        """blocks_50_50 split: LoRы мердджатся в первую половину DiT (cuda:0),
        значит cuda:0 должен включать ``loras_estimate``.
        """
        dit_gb, gemma_gb = 20.0, 7.5
        c0, c1 = _project(
            "blocks_50_50", dit_gb=dit_gb, gemma_gb=gemma_gb,
            components=COMPONENT_FOOTPRINT_GB,
        )
        loras_gb = COMPONENT_FOOTPRINT_GB["loras_estimate"]
        other_cuda0 = (
            COMPONENT_FOOTPRINT_GB["text_projection"]
            + COMPONENT_FOOTPRINT_GB["video_vae"]
            + COMPONENT_FOOTPRINT_GB["latent_upscaler"]
            + COMPONENT_FOOTPRINT_GB["sage_attention_scratch"]
        )
        expected_cuda0 = dit_gb / 2 + other_cuda0 + loras_gb
        self.assertAlmostEqual(c0, expected_cuda0, places=4)
        other_cuda1 = (
            COMPONENT_FOOTPRINT_GB["audio_vae"]
            + COMPONENT_FOOTPRINT_GB["sage_attention_scratch"]
        )
        expected_cuda1 = dit_gb / 2 + gemma_gb + other_cuda1
        self.assertAlmostEqual(c1, expected_cuda1, places=4)

    def test_blocks_30_70_loras_only_on_cuda0(self):
        """blocks_30_70: LoRы мердджатся в первую половину DiT (cuda:0 при 30%)."""
        c0, c1 = _project(
            "blocks_30_70", dit_gb=20.0, gemma_gb=7.5, components=COMPONENT_FOOTPRINT_GB,
        )
        other_cuda0 = (
            COMPONENT_FOOTPRINT_GB["text_projection"]
            + COMPONENT_FOOTPRINT_GB["video_vae"]
            + COMPONENT_FOOTPRINT_GB["latent_upscaler"]
            + COMPONENT_FOOTPRINT_GB["sage_attention_scratch"]
        )
        expected_cuda0_loras = 20.0 * 0.3 + other_cuda0 + COMPONENT_FOOTPRINT_GB["loras_estimate"]
        self.assertAlmostEqual(c0, expected_cuda0_loras, places=4)

    def test_single_cuda1_loras_on_cuda1(self):
        """single_cuda1: вся модель включая лоры — на cuda:1."""
        c0, c1 = _project(
            "single_cuda1", dit_gb=10.0, gemma_gb=7.5, components=COMPONENT_FOOTPRINT_GB,
        )
        other_cuda0 = (
            COMPONENT_FOOTPRINT_GB["text_projection"]
            + COMPONENT_FOOTPRINT_GB["video_vae"]
            + COMPONENT_FOOTPRINT_GB["latent_upscaler"]
            + COMPONENT_FOOTPRINT_GB["sage_attention_scratch"]
        )
        self.assertAlmostEqual(c0, other_cuda0, places=4)
        self.assertGreater(
            c1,
            10.0 + 7.5,
            "single_cuda1 должен включать и loras_estimate.",
        )

    def test_single_cuda0_loras_on_cuda0(self):
        """single_cuda0: вся модель включая лоры — на cuda:0."""
        c0, c1 = _project(
            "single_cuda0", dit_gb=10.0, gemma_gb=7.5, components=COMPONENT_FOOTPRINT_GB,
        )
        other_cuda1 = (
            COMPONENT_FOOTPRINT_GB["audio_vae"]
            + COMPONENT_FOOTPRINT_GB["sage_attention_scratch"]
        )
        self.assertAlmostEqual(c1, other_cuda1, places=4)
        self.assertGreater(c0, 10.0 + 7.5 + 2.55)

    def test_unknown_strategy_returns_zero_zero(self):
        """Defensive: unknown strategy → (0.0, 0.0) (без raise).

        Это контракт, который ``tests/test_diagnostic_pipeline_matches_
        memory_tracker.py`` тоже проверяет для standalone-CLI
        ``diagnostic.project_strategy``.
        """
        self.assertEqual(
            _project("blocks_70_30", 17.0, 7.5, COMPONENT_FOOTPRINT_GB),
            (0.0, 0.0),
        )
        self.assertEqual(
            _project("nope", 0.0, 0.0, COMPONENT_FOOTPRINT_GB),
            (0.0, 0.0),
        )

    def test_empty_components_double_safe(self):
        """Defensive: пустой components dict → projection даёт (dit+gemma,
        dit+other) без raise.

        Подтверждает, что ``_project`` использует ``components.get(..., 0.0)``
        вместо ``components[key]`` — безопасно при partial dict.
        """
        empty: dict[str, float] = {}
        # pipeline: c0 = gemma + 0, c1 = dit + 0
        c0, c1 = _project("pipeline", dit_gb=10.0, gemma_gb=7.5, components=empty)
        self.assertAlmostEqual(c0, 7.5, places=4)
        self.assertAlmostEqual(c1, 10.0, places=4)
        # blocks_50_50: c0 = dit/2 + 0, c1 = dit/2 + gemma + 0
        c0, c1 = _project("blocks_50_50", dit_gb=20.0, gemma_gb=7.5, components=empty)
        self.assertAlmostEqual(c0, 10.0, places=4)
        self.assertAlmostEqual(c1, 10.0 + 7.5, places=4)


# ════════════════════════════════════════════════════════════════════════════
# Block 3: classify_tensor — DiT tensor-name → category
# ════════════════════════════════════════════════════════════════════════════
class TestClassifyTensor(unittest.TestCase):
    """ДиТ-тензоры классифицируются по префиксу имени.

    Категории:
      - "block"   — transformer_blocks.{N}.* (44 блоков DiT)
      - "embed"   — time_embed / adaln / patchify_proj / proj_in / norm_in
      - "head"    — proj_out / norm_out
      - "other_diffusion" — другие .diffusion_model.* (не категоризировано)
      - "other"   — всё остальное (VAE / text encoder / внешнее)
    """

    def test_block_prefix_matches(self):
        """Основные 44 DiT-блока должны быть в категории 'block'."""
        for i in range(44):
            name = f"{DIFFUSION_BLOCK_PREFIX}{i}.attn.qkv.weight"
            with self.subTest(i=i):
                self.assertEqual(classify_tensor(name), "block")

    def test_embed_prefixes_match(self):
        """Embedding / AdaLN / proj_in / patchify_proj → 'embed'."""
        for prefix in DIFFUSION_EMBED_PREFIXES:
            self.assertTrue(
                prefix.startswith("model.diffusion_model."),
                f"Test fixture prefix должен начинаться с 'model.diffusion_model.': {prefix}",
            )
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    classify_tensor(f"{prefix}.weight"),
                    "embed",
                    f"prefix={prefix!r} должен categorise как 'embed'",
                )

    def test_head_proj_out(self):
        """proj_out.* → 'head'."""
        self.assertEqual(
            classify_tensor("model.diffusion_model.proj_out.weight"),
            "head",
        )

    def test_head_norm_out(self):
        """norm_out.* → 'head'."""
        self.assertEqual(
            classify_tensor("model.diffusion_model.norm_out.weight"),
            "head",
        )

    def test_other_diffusion(self):
        """Любое .diffusion_model.* не из известных категорий → 'other_diffusion'."""
        self.assertEqual(
            classify_tensor("model.diffusion_model.unexpected_submodule.weight"),
            "other_diffusion",
        )

    def test_vae_or_text_encoder_returns_other(self):
        """VAE / text encoder / фьюжн — outside diffusion_model scope."""
        for name in [
            "vae.encoder.conv.weight",
            "text_projection.weight",
            "model.text_model.layers.0.weight",
            "first_stage_model.encoder.weight",
        ]:
            with self.subTest(name=name):
                self.assertEqual(classify_tensor(name), "other")


# ════════════════════════════════════════════════════════════════════════════
# Block 4: _resolve_donor_device — UI-string → torch.device
# ════════════════════════════════════════════════════════════════════════════
class TestResolveDonorDevice(unittest.TestCase):
    """donor_device widget (auto/cuda:0/cuda:1/cpu) → torch.device."""

    PLACEHOLDER_PRIMARY = "cuda:0-placeholder"
    PLACEHOLDER_SECONDARY = "cuda:1-placeholder"
    PLACEHOLDER_CPU = "cpu"

    def test_auto_returns_secondary(self):
        """'auto' → secondary device (default cuda:1 в dual-GPU config)."""
        result = _resolve_donor_device(
            "auto",
            primary=self.PLACEHOLDER_PRIMARY,
            secondary=self.PLACEHOLDER_SECONDARY,
        )
        self.assertEqual(result, self.PLACEHOLDER_SECONDARY)

    def test_cuda0_returns_primary(self):
        """'cuda:0' → primary."""
        result = _resolve_donor_device(
            "cuda:0",
            primary=self.PLACEHOLDER_PRIMARY,
            secondary=self.PLACEHOLDER_SECONDARY,
        )
        self.assertEqual(result, self.PLACEHOLDER_PRIMARY)

    def test_cuda1_returns_secondary(self):
        """'cuda:1' → secondary."""
        result = _resolve_donor_device(
            "cuda:1",
            primary=self.PLACEHOLDER_PRIMARY,
            secondary=self.PLACEHOLDER_SECONDARY,
        )
        self.assertEqual(result, self.PLACEHOLDER_SECONDARY)

    def test_cpu_returns_cpu(self):
        """'cpu' → cpu device placeholder."""
        result = _resolve_donor_device(
            "cpu",
            primary=self.PLACEHOLDER_PRIMARY,
            secondary=self.PLACEHOLDER_SECONDARY,
        )
        self.assertTrue(
            str(result).startswith("cpu"),
            f"'cpu' spec должен вернуть cpu-like device, got: {result!r}",
        )

    def test_empty_spec_defaults_to_secondary_via_swallow(self):
        """Пустая строка / неизвестный spec для ``_resolve_donor_device``."""
        for spec in ("", "  ", "unknown"):
            with self.subTest(spec=spec):
                try:
                    result = _resolve_donor_device(
                        spec,
                        primary=self.PLACEHOLDER_PRIMARY,
                        secondary=self.PLACEHOLDER_SECONDARY,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.fail(
                        f"_resolve_donor_device({spec!r}) raised: {exc!r} — "
                        "design expects silent fallback to secondary."
                    )
                self.assertIsNotNone(result)

    def test_resolve_donor_device_module_export(self):
        """gguf_split экспортирует ``_resolve_donor_device``."""
        self.assertTrue(hasattr(gguf_split, "_resolve_donor_device"))
        self.assertTrue(callable(gguf_split._resolve_donor_device))


if __name__ == "__main__":
    unittest.main(verbosity=2)
