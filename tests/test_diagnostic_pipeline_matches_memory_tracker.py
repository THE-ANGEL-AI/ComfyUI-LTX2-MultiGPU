"""Regression test: drift-сценарий ``scripts/diagnostic.py::project_strategy``
↔ ``core/memory_tracker._project``.

История бага (закрыт этим тестом):
  - v0.2.1 (FIX MED-4): в ``core/memory_tracker._project("pipeline")`` добавлен
    ``loras_estimate`` на стороне cuda:1.
  - ``scripts/diagnostic.py::project_strategy`` имел stale-копию math ДО этого
    фикса — расхождение на ~2.55 GB silent. Standalone-CLI показывал
    pipeline как 'OK' на edge-budget конфигурациях с LoRами, где in-package
    бы показал OOM.
  - Никакие tests не сравнивали standalone-CLI vs in-package. Существующий
    ``tests/test_memory_tracker.py::TestProjectPerStrategy`` MIRROR-ил math
    (nested closure в ``estimate_vram_budget`` был недоступен для import →
    tests копировали logic вручную), что само по себе позволяет drifted
    copies проходить под тестом.

  - Этот тест использует последний v0.2.2-pre+ refactor: ``memory_tracker._project``
    поднят на module-level (см. core/memory_tracker.py), making directly importable.
    Теперь мы сравниваем ИСТИННЫЕ две реализации на shared fixtures → drift
    ловится немедленно с понятным (strategy, fixture) subTest-именем.

Запуск:
    python -m unittest tests.test_diagnostic_pipeline_matches_memory_tracker -v

Pure-stdlib unittest. NO torch, NO ComfyUI runtime required (diagnostic.py
использует try/except для gguf/safetensors, поэтому модуль импортируется без
реальных CUDA / torch).
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBMODULE_ROOT = HERE.parent
# Super-project layout (canonical):
#     <repo>/custom_nodes/<submodule>/tests/<this_test>.py
#     <repo>/scripts/diagnostic.py
# ⇒ диагностика лежит на 2 уровня выше submodule root.
SUPERPROJECT_ROOT = SUBMODULE_ROOT.parent.parent
SCRIPTS_DIR = SUPERPROJECT_ROOT / "scripts"
# Подключаем scripts/ к sys.path чтобы ``import diagnostic`` резолвился
# в standalone-CLI ``scripts/diagnostic.py``.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Подключаем submodule root к sys.path чтобы ``from core.memory_tracker …``
# работало. CI обычно запускает из submodule root (cwd), но explicit insert
# — defensive на случай запуска из произвольной локации.
if str(SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMODULE_ROOT))

from core.memory_tracker import COMPONENT_FOOTPRINT_GB, _project  # noqa: E402
import diagnostic  # type: ignore[import-not-found]  # noqa: E402


ALL_STRATEGIES: tuple[str, ...] = (
    "blocks_50_50",
    "blocks_30_70",
    "pipeline",
    "single_cuda0",
    "single_cuda1",
)

# Shared fixtures: покрывают Kaggle T4×2 / RTX 4090×2 / asymmetric setups.
SHARED_FIXTURES: list[dict] = [
    {"name": "kaggle_t4_q4_default", "dit_gb": 17.0, "gemma_gb": 7.5,
     "cap0_gb": 15.0, "cap1_gb": 15.0},
    {"name": "kaggle_t4_q5_inplace", "dit_gb": 21.0, "gemma_gb": 7.5,
     "cap0_gb": 15.0, "cap1_gb": 15.0},
    {"name": "rtx_4090_pair",       "dit_gb": 17.0, "gemma_gb": 7.5,
     "cap0_gb": 24.0, "cap1_gb": 24.0},
    # diagnostic.project_strategy принимает gemma_gb как float | None;
    # memory_tracker._project нормализует через 0.0 (None не валиден).
    {"name": "no_gemma",            "dit_gb": 17.0, "gemma_gb": None,
     "cap0_gb": 15.0, "cap1_gb": 15.0},
    {"name": "asymmetric_caps",     "dit_gb": 22.0, "gemma_gb": 7.5,
     "cap0_gb":  8.0, "cap1_gb": 24.0},
]


class TestProjectStrategyMatches(unittest.TestCase):
    """Cross-check ``diagnostic.project_strategy(...)`` ≡ ``memory_tracker._project(...)``.

    Если кто-то меняет одну реализацию и забывает другую — этот test падает
    с конкретным ``(strategy, fixture)`` subTest-именем.
    """

    def test_pipeline_matches_across_all_fixtures(self):
        """Главный regressor: pipeline strategy (drift-сценарий, отлавливает
        отсутствие ``loras_estimate`` на cuda:1). Должна совпадать на каждой
        фикстуре.
        """
        for fixture in SHARED_FIXTURES:
            with self.subTest(fixture=fixture["name"]):
                diag_result = diagnostic.project_strategy(
                    "pipeline",
                    fixture["dit_gb"],
                    fixture["gemma_gb"],
                    fixture["cap0_gb"],
                    fixture["cap1_gb"],
                    COMPONENT_FOOTPRINT_GB,
                )
                mt_c0, mt_c1 = _project(
                    "pipeline",
                    fixture["dit_gb"],
                    fixture["gemma_gb"] or 0.0,
                    COMPONENT_FOOTPRINT_GB,
                )

                self.assertAlmostEqual(
                    diag_result["cuda0_load_gb"], round(mt_c0, 3), places=3,
                    msg=f"pipeline cuda0 mismatch on fixture {fixture['name']!r}",
                )
                self.assertAlmostEqual(
                    diag_result["cuda1_load_gb"], round(mt_c1, 3), places=3,
                    msg=f"pipeline cuda1 mismatch on fixture {fixture['name']!r}",
                )

    def test_all_strategies_match_default_fixture(self):
        """Универсальный regressor: каждая из 5 стратегий должна совпадать
        на default fixture. Drift может возникнуть в ЛЮБОЙ стратегии, не
        только pipeline (например, кто-то добавит новую ``blocks_70_30``).
        """
        fixture = SHARED_FIXTURES[0]  # kaggle t4 q4 default
        for strategy in ALL_STRATEGIES:
            with self.subTest(strategy=strategy):
                diag_result = diagnostic.project_strategy(
                    strategy,
                    fixture["dit_gb"],
                    fixture["gemma_gb"],
                    fixture["cap0_gb"],
                    fixture["cap1_gb"],
                    COMPONENT_FOOTPRINT_GB,
                )
                mt_c0, mt_c1 = _project(
                    strategy,
                    fixture["dit_gb"],
                    fixture["gemma_gb"] or 0.0,
                    COMPONENT_FOOTPRINT_GB,
                )
                self.assertAlmostEqual(
                    diag_result["cuda0_load_gb"], round(mt_c0, 3), places=3,
                    msg=f"{strategy} cuda0 mismatch on default fixture",
                )
                self.assertAlmostEqual(
                    diag_result["cuda1_load_gb"], round(mt_c1, 3), places=3,
                    msg=f"{strategy} cuda1 mismatch on default fixture",
                )

    def test_unknown_strategy_behavior_consistent(self):
        """Documenting-the-asymmetry test для unknown strategy.

        Текущее поведение (на момент v0.2.2-pre+):
          - ``diagnostic.project_strategy(unknown)``: raise ``ValueError``
            (явная защита от drift: caller знает что strategy не поддержана).
          - ``memory_tracker._project(unknown)``: return ``(0.0, 0.0)``
            (тихий fallback — backward-compatible с pre-v0.2.1 кодом,
             который мог случайно передавать unknown strategy и не получить
             exception, только ``0.0`` projection).

        Lock обе стороны на текущее поведение → если кто-то изменит одно из
        них (в любую сторону), тест поймает drift assumption. Cross-check
        не enforcement'ит IDENTICAL behavior (намеренно, asymmetry описана
        в комментариях), но lock'ит обе реализации на их собственный current
        contract.
        """
        unknown = "blocks_70_30"  # hypothetical future addition w/o branch

        # diagnostic поднимает ValueError
        with self.assertRaises(ValueError) as exc_ctx:
            diagnostic.project_strategy(
                unknown, 17.0, 7.5, 15.0, 15.0, COMPONENT_FOOTPRINT_GB,
            )
        self.assertIn(
            "unknown strategy", str(exc_ctx.exception).lower(),
            "diagnostic должен явно сообщить об unknown strategy в сообщении.",
        )

        # memory_tracker возвращает (0.0, 0.0)
        self.assertEqual(
            _project(unknown, 17.0, 7.5, COMPONENT_FOOTPRINT_GB),
            (0.0, 0.0),
        )

    def test_diagnostic_STRATEGIES_matches_memory_tracker_branches(self):
        """Sanity: количество strategy в ``diagnostic.py::STRATEGIES`` должно
        совпадать с известными ветками ``memory_tracker._project``. Иначе
        одна из реализаций поддерживает strategy, которую другая не знает
        (orphan direction of drift).
        """
        known_branches = set(ALL_STRATEGIES)
        self.assertEqual(
            set(diagnostic.STRATEGIES), known_branches,
            "diagnostic.STRATEGIES должна покрывать ровно 5 known branches.\n"
            "Если добавляете новую strategy в diagnostic.py — добавьте branch "
            "и в memory_tracker._project(), и обновите known_branches здесь.",
        )

    def test_diagnostic_COMPONENT_FOOTPRINT_GB_matches(self):
        """Cross-check: ключи footprint dict должны быть идентичны (HIGH-1 fix
        в v0.2.1 synchronised ``sage_attention_scratch`` key; защита от
        recurrence).
        """
        self.assertEqual(
            set(diagnostic.COMPONENT_FOOTPRINT_GB.keys()),
            set(COMPONENT_FOOTPRINT_GB.keys()),
            "diagnostic.COMPONENT_FOOTPRINT_GB keys должны совпадать с "
            "memory_tracker.COMPONENT_FOOTPRINT_GB keys. Если добавляете "
            "новый компонент — добавьте в ОБА. См. CHANGELOG v0.2.1 HIGH-1.",
        )
        for k in COMPONENT_FOOTPRINT_GB:
            self.assertAlmostEqual(
                diagnostic.COMPONENT_FOOTPRINT_GB[k],
                COMPONENT_FOOTPRINT_GB[k],
                places=4,
                msg=f"footprint value differs for key {k!r}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
