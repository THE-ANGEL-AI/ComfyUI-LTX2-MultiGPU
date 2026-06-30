"""Regression test for core.gguf_split._split_blocks_indices.

Pure-stdlib unittest (no pytest required). Run via:
    python -m unittest tests.test_gguf_split_blocks_indices -v
    # or directly:
    python tests/test_gguf_split_blocks_indices.py

REGRESSION TARGET (issue post-v0.2.0 audit):
    ``_split_blocks_indices("blocks_30_70")`` previously returned ``(14,)``
    which yielded an actual 32/68 split (14 blocks on primary, 30 on donor).
    The 30/70 naming was misleading the memory_tracker's heuristic and
    could trigger OOM at runtime where the tracker had predicted OK.
    Fixed: return ``(13,)`` → real 30/70 split (13 blocks on primary,
    31 on donor = ~30%/~70% of LTX2_DIT_BLOCK_COUNT=44).
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Submodule layout: tests/ sits next to core/, nodes.py, __init__.py.
# Adding the submodule root to sys.path lets ``from core.gguf_split …`` work
# both in CI (where ci.yml runs from submodule root) and locally.
SUBMODULE_ROOT = HERE.parent
if str(SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMODULE_ROOT))

from core import gguf_split as _gguf_split  # noqa: E402
from core.gguf_split import (  # noqa: E402
    LTX2_DIT_BLOCK_COUNT,
    STRATEGIES,
    _split_blocks_indices,
)


class TestSplitBlocksIndices(unittest.TestCase):
    """Регрессионные проверки для _split_blocks_indices."""

    def test_blocks_50_50_returns_22(self):
        """50/50 split должен давать split_idx=22 (22 блока на device0, 22 на device1)."""
        result = _split_blocks_indices("blocks_50_50")
        self.assertEqual(result, (22,))
        # Geometric invariant: split_idx + (N - split_idx) == N (zero loss).
        self.assertEqual(
            result[0] + (LTX2_DIT_BLOCK_COUNT - result[0]),
            LTX2_DIT_BLOCK_COUNT,
        )

    def test_blocks_30_70_returns_13(self):
        """🚨 REGRESSION GUARD: 30/70 split должен давать split_idx=13.

        13 блоков на device0 (≈30%), 31 на device1 (≈70%).
        Historically returned (14,) — that was a 32/68 split, contradicting
        the strategy name. memory_tracker's projection heuristic computed
        budget for 30/70, runtime got 32/68 → OOM on edge budgets.
        """
        result = _split_blocks_indices("blocks_30_70")
        self.assertEqual(result, (13,))
        device0_share = result[0] / LTX2_DIT_BLOCK_COUNT
        device1_share = (LTX2_DIT_BLOCK_COUNT - result[0]) / LTX2_DIT_BLOCK_COUNT
        self.assertAlmostEqual(device0_share, 0.30, places=2)
        self.assertAlmostEqual(device1_share, 0.70, places=2)

    def test_pipeline_returns_empty_split(self):
        """pipeline strategy = whole-model move → нет split marker."""
        self.assertEqual(_split_blocks_indices("pipeline"), ())

    def test_single_cuda0_returns_empty_split(self):
        """single_cuda0 = whole-model move на primary → нет split."""
        self.assertEqual(_split_blocks_indices("single_cuda0"), ())

    def test_single_cuda1_returns_empty_split(self):
        """single_cuda1 = whole-model move на secondary → нет split."""
        self.assertEqual(_split_blocks_indices("single_cuda1"), ())

    def test_unknown_strategy_returns_empty_defensively(self):
        """Defensive: любая незнакомая strategy возвращает () без exceptions."""
        for unknown in ("blocks_70_30", "", "nonexistent", "BLOCKS_30_70", "blocks_3030"):
            with self.subTest(unknown=unknown):
                self.assertEqual(_split_blocks_indices(unknown), ())

    def test_all_STRATEGIES_resolvable(self):
        """Каждая strategy из публичного STRATEGIES tuple обязана вернуть tuple
        без exceptions и без невалидных индексов.

        ⚠️ Caveat (false-negative trap): этот тест НЕ ловит ситуацию, когда
        кто-то добавил новую split-bearing strategy в ``STRATEGIES`` tuple
        (например, ``blocks_70_30``) и ЗАБЫЛ добавить соответствующую ветку в
        ``_split_blocks_indices`` — функция дефолтно возвращает ``()`` без
        raise, тест passes, а split-mode silently no-op'ит → runtime OOM на
        user-machine. Дополнительная защита — реальная runtime-validation в
        ``apply_strategy`` (визуальный WARN + fallback на primary_dev) и
        review'nит это изменение руками. Не полагайтесь только на unittest.
        """
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy):
                result = _split_blocks_indices(strategy)
                self.assertIsInstance(result, tuple)
                for idx in result:
                    self.assertIsInstance(idx, int)
                    self.assertGreaterEqual(idx, 0)
                    self.assertLessEqual(idx, LTX2_DIT_BLOCK_COUNT)

    def test_strategies_provider_consistent(self):
        """``STRATEGIES`` — tuple из ровно 5 стратегий (current spec)."""
        self.assertEqual(
            set(STRATEGIES),
            {"blocks_50_50", "blocks_30_70", "pipeline", "single_cuda0", "single_cuda1"},
        )

    def test_block_count_constant_matches_spec(self):
        """LTX2_DIT_BLOCK_COUNT == 44 (см. MODEL_FACTS §3). Если кто-то поменяет
        константу — тесты выше зафейлятся с понятным message."""
        self.assertEqual(LTX2_DIT_BLOCK_COUNT, 44)

    def test_block_split_indices_module_export(self):
        """core.gguf_split экспортирует _split_blocks_indices (для тестов и для
        возможного reuse из nodes.py во 2-й итерации)."""
        self.assertTrue(hasattr(_gguf_split, "_split_blocks_indices"))
        self.assertTrue(callable(_gguf_split._split_blocks_indices))


if __name__ == "__main__":
    unittest.main(verbosity=2)
