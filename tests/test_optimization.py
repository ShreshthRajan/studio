"""Tests for the optimization loop (Step 6)."""

import pytest
from unittest.mock import MagicMock

from envaudit.optimization.augmenter import (
    TestAugmenter as Augmenter,
    TestAugmentation as AugmentationEntry,
    AugmentationResult,
)
from envaudit.optimization.loop import (
    run_optimization_loop,
    _estimate_eqs_boost,
    _run_iteration,
    OptimizationResult,
)
from envaudit.scoring.composite import CompositeResult
from envaudit.llm.claude_client import ClaudeResponse


# ─── Helpers ────────────────────────────────────────────────────────────

def _mock_response(augmentations):
    return ClaudeResponse(
        content="",
        parsed_json={"augmentations": augmentations},
        input_tokens=1000, output_tokens=500,
        cost_usd=0.02, latency_seconds=3.0,
        model="claude-sonnet-4-20250514",
    )


def _make_fix_task(iid, eqs=0.5):
    return CompositeResult(
        instance_id=iid, eqs=eqs, verdict="FIX",
        dimension_scores={"verifier": 0.7, "exploit": 0.3},
        weaknesses=[], weights={},
    )


def _make_task_data(iid, n_exploits=2):
    return {
        "problem_statement": "Fix the bug in module X.",
        "test_patch": "diff --git a/tests/test.py b/tests/test.py\n+def test_x():\n+    assert x() == 1",
        "gold_patch": "diff --git a/module.py b/module.py\n+    return correct_value",
        "passed_exploits": [
            {
                "patch_id": f"{iid}_exploit_{i}",
                "patch_text": f"--- a/module.py\n+++ b/module.py\n+    return hacked_{i}",
                "strategy_name": f"exploit_{i}",
                "strategy_description": f"Exploit strategy {i}",
            }
            for i in range(n_exploits)
        ],
    }


# ─── Augmenter Tests ───────────────────────────────────────────────────

class TestAugmenterModule:
    def test_generates_augmentations(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_response([
            {
                "test_code": "def test_edge_case(self):\n    assert x(0) == 0",
                "rationale": "Exploit 0 hardcodes the return value, this tests a different input",
                "targets": ["exploit_0"],
                "confidence": 0.85,
            },
        ])

        augmenter = Augmenter(mock_client)
        result = augmenter.augment_task(
            instance_id="test__task-1",
            problem_statement="Fix the bug",
            existing_test_patch="def test_x(): assert x(1) == 1",
            gold_patch="return a + b",
            passed_exploits=[{
                "patch_id": "test__task-1_exploit_0",
                "patch_text": "return 1",
                "strategy_name": "hardcode",
                "strategy_description": "Hardcodes return value",
            }],
        )

        assert len(result.augmentations) == 1
        assert "test_edge_case" in result.augmentations[0].test_code
        assert result.augmentations[0].confidence == 0.85
        assert result.n_exploits_targeted == 1

    def test_handles_no_exploits(self):
        augmenter = Augmenter(MagicMock())
        result = augmenter.augment_task("task", "prob", "test", "gold", [])
        assert len(result.augmentations) == 0
        assert result.total_cost_usd == 0.0

    def test_handles_failed_parse(self):
        mock_client = MagicMock()
        mock_client.query.return_value = ClaudeResponse(
            content="I cannot generate tests",
            parsed_json=None,
            input_tokens=500, output_tokens=100,
            cost_usd=0.005, latency_seconds=1.0,
            model="test",
        )

        augmenter = Augmenter(mock_client)
        result = augmenter.augment_task("task", "prob", "test", "gold", [
            {"patch_id": "e0", "patch_text": "p", "strategy_name": "s", "strategy_description": "d"},
        ])
        assert len(result.augmentations) == 0

    def test_skips_empty_test_code(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_response([
            {"test_code": "", "rationale": "empty", "targets": [], "confidence": 0.5},
            {"test_code": "def test_real(): pass", "rationale": "real", "targets": ["exploit_0"], "confidence": 0.8},
        ])

        augmenter = Augmenter(mock_client)
        result = augmenter.augment_task("task", "prob", "test", "gold", [
            {"patch_id": "e0", "patch_text": "p", "strategy_name": "s", "strategy_description": "d"},
        ])
        assert len(result.augmentations) == 1  # Skipped the empty one


# ─── EQS Boost Estimation Tests ────────────────────────────────────────

class TestEstimateBoost:
    def test_no_augmentations_no_boost(self):
        result = AugmentationResult("t", [], 0.0, 2)
        assert _estimate_eqs_boost(result, 0.5) == 0.0

    def test_high_confidence_full_coverage(self):
        result = AugmentationResult(
            "t",
            [AugmentationEntry("t", "code", ["e0", "e1"], "r", True, 0.9, 0.01)],
            0.01, 2,
        )
        boost = _estimate_eqs_boost(result, 0.5)
        assert boost > 0.0
        assert boost <= 0.15  # Max conservative boost

    def test_low_confidence_less_boost(self):
        high_conf = AugmentationResult(
            "t",
            [AugmentationEntry("t", "code", ["e0"], "r", True, 0.9, 0.01)],
            0.01, 1,
        )
        low_conf = AugmentationResult(
            "t",
            [AugmentationEntry("t", "code", ["e0"], "r", True, 0.3, 0.01)],
            0.01, 1,
        )
        assert _estimate_eqs_boost(high_conf, 0.5) > _estimate_eqs_boost(low_conf, 0.5)

    def test_partial_coverage_less_boost(self):
        full = AugmentationResult(
            "t",
            [AugmentationEntry("t", "code", ["e0", "e1"], "r", True, 0.8, 0.01)],
            0.01, 2,
        )
        partial = AugmentationResult(
            "t",
            [AugmentationEntry("t", "code", ["e0"], "r", True, 0.8, 0.01)],
            0.01, 2,
        )
        assert _estimate_eqs_boost(full, 0.5) > _estimate_eqs_boost(partial, 0.5)


# ─── Optimization Loop Tests ──────────────────────────────────────────

class TestOptimizationLoop:
    def test_single_iteration(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_response([
            {"test_code": "def test_new(): assert True", "rationale": "blocks exploit",
             "targets": ["exploit_0"], "confidence": 0.8},
        ])

        fix_tasks = {"task1": _make_fix_task("task1", eqs=0.5)}
        task_data = {"task1": _make_task_data("task1", n_exploits=1)}

        result = run_optimization_loop(
            fix_tasks, task_data,
            Augmenter(mock_client),
            max_iterations=1,
        )

        assert result.total_iterations == 1
        assert result.initial_mean_eqs == 0.5
        assert result.final_mean_eqs >= 0.5  # Should improve or stay same
        assert len(result.convergence_curve) == 2  # Initial + 1 iteration

    def test_converges_when_no_improvement(self):
        mock_client = MagicMock()
        # Return empty augmentations → no improvement → converge
        mock_client.query.return_value = _mock_response([])

        fix_tasks = {"task1": _make_fix_task("task1", eqs=0.5)}
        task_data = {"task1": _make_task_data("task1")}

        result = run_optimization_loop(
            fix_tasks, task_data,
            Augmenter(mock_client),
            max_iterations=3,
        )

        assert result.converged
        assert result.total_iterations == 1  # Stops after first iteration with no improvement

    def test_multiple_tasks(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_response([
            {"test_code": "def test_x(): pass", "rationale": "r",
             "targets": ["exploit_0"], "confidence": 0.7},
        ])

        fix_tasks = {
            "task1": _make_fix_task("task1", eqs=0.45),
            "task2": _make_fix_task("task2", eqs=0.55),
        }
        task_data = {
            "task1": _make_task_data("task1"),
            "task2": _make_task_data("task2"),
        }

        result = run_optimization_loop(
            fix_tasks, task_data,
            Augmenter(mock_client),
            max_iterations=1,
        )

        assert result.total_iterations == 1
        assert len(result.iterations[0].per_task) == 2

    def test_empty_fix_tasks(self):
        result = run_optimization_loop(
            {}, {}, Augmenter(MagicMock()), max_iterations=3,
        )
        assert result.total_iterations == 0
        assert result.converged

    def test_tasks_upgraded_counted(self):
        mock_client = MagicMock()
        # High confidence augmentation targeting all exploits
        mock_client.query.return_value = _mock_response([
            {"test_code": "def test_blocking(): assert True", "rationale": "blocks all",
             "targets": ["exploit_0", "exploit_1"], "confidence": 0.95},
        ])

        # Task at 0.65 — close to KEEP threshold (0.70)
        fix_tasks = {"task1": _make_fix_task("task1", eqs=0.65)}
        task_data = {"task1": _make_task_data("task1", n_exploits=2)}

        result = run_optimization_loop(
            fix_tasks, task_data,
            Augmenter(mock_client),
            max_iterations=1,
        )

        # With ~0.14 boost, 0.65 + 0.14 = 0.79 > 0.70 → should be upgraded
        assert result.tasks_upgraded_total >= 1
