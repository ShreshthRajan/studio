"""Tests for difficulty profiling (Step 4)."""

import pytest
from envaudit.scoring.difficulty import (
    profile_difficulty,
    compute_solve_rates,
    load_solve_rates_from_results,
    DifficultyProfile,
)


class TestProfileDifficulty:
    def test_optimal_difficulty(self):
        """p=0.5 should give maximum gradient signal."""
        d = profile_difficulty("task1", 0.5, n_models=50)
        assert d.gradient_signal == 0.25  # max p(1-p)
        assert d.difficulty_tier == "medium"
        assert d.flag is None

    def test_too_hard(self):
        d = profile_difficulty("task2", 0.02, n_models=50)
        assert d.difficulty_tier == "too_hard"
        assert d.flag == "too_hard"
        assert d.gradient_signal < 0.05

    def test_too_easy(self):
        d = profile_difficulty("task3", 0.98, n_models=50)
        assert d.difficulty_tier == "too_easy"
        assert d.flag == "too_easy"
        assert d.gradient_signal < 0.05

    def test_hard_but_not_too_hard(self):
        d = profile_difficulty("task4", 0.10, n_models=50)
        assert d.difficulty_tier == "hard"
        assert d.flag is None

    def test_easy_but_not_too_easy(self):
        d = profile_difficulty("task5", 0.90, n_models=50)
        assert d.difficulty_tier == "easy"
        assert d.flag is None

    def test_clamps_solve_rate(self):
        d = profile_difficulty("task6", 1.5, n_models=10)
        assert d.solve_rate == 1.0
        d2 = profile_difficulty("task7", -0.3, n_models=10)
        assert d2.solve_rate == 0.0

    def test_zero_solve_rate(self):
        d = profile_difficulty("task8", 0.0, n_models=50)
        assert d.gradient_signal == 0.0
        assert d.difficulty_tier == "too_hard"

    def test_gradient_symmetry(self):
        """p and (1-p) should give same gradient signal."""
        d1 = profile_difficulty("a", 0.3, n_models=10)
        d2 = profile_difficulty("b", 0.7, n_models=10)
        assert abs(d1.gradient_signal - d2.gradient_signal) < 0.001


class TestComputeSolveRates:
    def test_basic(self):
        resolved_sets = [
            {"task1", "task2"},
            {"task1", "task3"},
            {"task1"},
        ]
        rates = compute_solve_rates(resolved_sets)
        assert rates["task1"] == 1.0      # All 3 models solved it
        assert rates["task2"] == 1 / 3    # 1 of 3
        assert rates["task3"] == 1 / 3

    def test_with_target_filter(self):
        resolved_sets = [
            {"task1", "task2", "task99"},
            {"task1"},
        ]
        rates = compute_solve_rates(resolved_sets, target_instance_ids={"task1", "task2"})
        assert "task1" in rates
        assert "task2" in rates
        assert "task99" not in rates

    def test_unsolved_tasks_included(self):
        """Tasks in target set that no model solved should have rate 0."""
        resolved_sets = [{"task1"}]
        rates = compute_solve_rates(resolved_sets, target_instance_ids={"task1", "task2"})
        assert rates["task2"] == 0.0

    def test_empty_submissions(self):
        rates = compute_solve_rates([])
        assert rates == {}

    def test_all_models_solve_all(self):
        resolved_sets = [{"t1", "t2"}, {"t1", "t2"}, {"t1", "t2"}]
        rates = compute_solve_rates(resolved_sets)
        assert rates["t1"] == 1.0
        assert rates["t2"] == 1.0


class TestLoadSolveRatesFromResults:
    def test_basic(self):
        results = [
            {"resolved": ["task1", "task2"]},
            {"resolved": ["task1"]},
        ]
        rates = load_solve_rates_from_results(results)
        assert rates["task1"] == 1.0
        assert rates["task2"] == 0.5

    def test_with_target(self):
        results = [
            {"resolved": ["task1", "task2", "task3"]},
        ]
        rates = load_solve_rates_from_results(results, target_instance_ids={"task1"})
        assert "task1" in rates
        assert "task2" not in rates

    def test_empty_resolved(self):
        results = [{"resolved": []}, {"resolved": []}]
        rates = load_solve_rates_from_results(results, target_instance_ids={"task1"})
        assert rates["task1"] == 0.0
