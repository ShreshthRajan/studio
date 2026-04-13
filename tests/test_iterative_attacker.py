"""Tests for the Iterative Adversarial Attacker (Rounds 2+3)."""

import pytest
from unittest.mock import MagicMock
from envaudit.agents.iterative_attacker import (
    IterativeAttacker,
    RoundResult,
    IterativeAttackResult,
    _parse_strategies,
)
from envaudit.agents.hackability import ExploitStrategy
from envaudit.agents.base import TaskData
from envaudit.llm.claude_client import ClaudeResponse


# ─── Test Data ──────────────────────────────────────────────────────────

SAMPLE_TASK = TaskData(
    instance_id="django/django-11133",
    repo="django/django",
    problem_statement="Fix HttpResponse to handle memoryview content correctly.",
    test_patch="""diff --git a/tests/test_response.py b/tests/test_response.py
--- a/tests/test_response.py
+++ b/tests/test_response.py
@@ -1,3 +1,8 @@
+from django.http import HttpResponse
+
+class TestMemoryView(TestCase):
+    def test_memoryview_content(self):
+        response = HttpResponse(memoryview(b"hello"))
+        self.assertEqual(response.content, b"hello")
""",
    gold_patch="""diff --git a/django/http/response.py b/django/http/response.py
--- a/django/http/response.py
+++ b/django/http/response.py
@@ -228,6 +228,8 @@
     def content(self, value):
+        if isinstance(value, memoryview):
+            value = bytes(value)
         self._container = [value]
""",
    fail_to_pass=["tests/test_response.py::TestMemoryView::test_memoryview_content"],
    pass_to_pass=["tests/test_response.py::TestBasic::test_basic"],
    is_verified=True,
)

ROUND1_STRATEGIES = [
    {
        "name": "hardcoded_memoryview_literal",
        "category": "test_suite_exploitation",
        "confidence": 9,
        "description": "Hardcode b'hello' when memoryview detected",
    },
    {
        "name": "memoryview_string_conversion",
        "category": "solution_degradation",
        "confidence": 7,
        "description": "Convert memoryview to string instead of bytes",
    },
]


def _mock_claude_response(strategies_json):
    """Create a mock ClaudeResponse with given strategies."""
    return ClaudeResponse(
        content="",
        parsed_json={"strategies": strategies_json},
        input_tokens=1000,
        output_tokens=500,
        cost_usd=0.02,
        latency_seconds=2.0,
        model="claude-sonnet-4-20250514",
    )


def _mock_empty_response():
    """Claude returns no valid strategies."""
    return ClaudeResponse(
        content="I couldn't generate exploits",
        parsed_json=None,
        input_tokens=500,
        output_tokens=100,
        cost_usd=0.005,
        latency_seconds=1.0,
        model="claude-sonnet-4-20250514",
    )


# ─── Parse Strategies Tests ─────────────────────────────────────────────

class TestParseStrategies:
    def test_parses_valid_response(self):
        response = _mock_claude_response([
            {"name": "test", "category": "test_suite_exploitation",
             "description": "desc", "exploit_patch": "patch", "confidence": 8, "reasoning": "r"},
        ])
        strategies = _parse_strategies(response, round_num=2)
        assert len(strategies) == 1
        assert strategies[0].name == "test"
        assert strategies[0].confidence == 8

    def test_handles_empty_response(self):
        response = _mock_empty_response()
        strategies = _parse_strategies(response, round_num=2)
        assert len(strategies) == 0

    def test_clamps_confidence_to_10(self):
        response = _mock_claude_response([
            {"name": "x", "category": "x", "description": "x",
             "exploit_patch": "x", "confidence": 15, "reasoning": "x"},
        ])
        strategies = _parse_strategies(response, round_num=2)
        assert strategies[0].confidence == 10

    def test_handles_missing_fields(self):
        response = _mock_claude_response([{"name": "minimal"}])
        strategies = _parse_strategies(response, round_num=2)
        assert len(strategies) == 1
        assert strategies[0].category == "unknown"
        assert strategies[0].confidence == 5


# ─── Round 2 Tests ──────────────────────────────────────────────────────

class TestRound2:
    def test_generates_strategies(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_claude_response([
            {"name": "new_approach", "category": "context_exploitation",
             "description": "tries different file", "exploit_patch": "--- a/x\n+++ b/x",
             "confidence": 7, "reasoning": "avoids Round 1 failure"},
        ])

        attacker = IterativeAttacker(mock_client)
        result = attacker.attack_round2(SAMPLE_TASK, ROUND1_STRATEGIES, {0: False, 1: False})

        assert result.round_num == 2
        assert len(result.strategies) == 1
        assert result.strategies[0].name == "new_approach"
        assert result.cost_usd == 0.02

    def test_includes_round1_summary_in_prompt(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_claude_response([])

        attacker = IterativeAttacker(mock_client)
        attacker.attack_round2(SAMPLE_TASK, ROUND1_STRATEGIES, {0: False, 1: False})

        call_args = mock_client.query.call_args
        prompt = call_args.kwargs.get("prompt", call_args[0][0] if call_args[0] else "")
        assert "hardcoded_memoryview_literal" in prompt
        assert "FAILED" in prompt

    def test_docker_results_shown_in_prompt(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_claude_response([])

        attacker = IterativeAttacker(mock_client)
        # One passed, one failed
        attacker.attack_round2(SAMPLE_TASK, ROUND1_STRATEGIES, {0: True, 1: False})

        prompt = mock_client.query.call_args.kwargs.get("prompt") or mock_client.query.call_args[0][0]
        assert "PASSED" in prompt
        assert "FAILED" in prompt

    def test_handles_no_docker_data(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_claude_response([])

        attacker = IterativeAttacker(mock_client)
        result = attacker.attack_round2(SAMPLE_TASK, ROUND1_STRATEGIES, None)
        assert result.round_num == 2


# ─── Round 3 Tests ──────────────────────────────────────────────────────

class TestRound3:
    def test_generates_almost_correct_patches(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_claude_response([
            {"name": "off_by_one_bytes", "category": "almost_correct",
             "description": "converts to bytes but misses edge case",
             "exploit_patch": "--- a/django/http/response.py\n+++ b/django/http/response.py",
             "confidence": 6, "reasoning": "subtle error in type check"},
        ])

        attacker = IterativeAttacker(mock_client)
        result = attacker.attack_round3(SAMPLE_TASK)

        assert result.round_num == 3
        assert len(result.strategies) == 1
        assert result.strategies[0].category == "almost_correct"

    def test_uses_full_gold_patch(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_claude_response([])

        attacker = IterativeAttacker(mock_client)
        attacker.attack_round3(SAMPLE_TASK)

        prompt = mock_client.query.call_args.kwargs.get("prompt") or mock_client.query.call_args[0][0]
        # Round 3 should include the gold patch for modification
        assert "memoryview" in prompt or "isinstance" in prompt


# ─── Full Pipeline Tests ────────────────────────────────────────────────

class TestRunAllRounds:
    def test_runs_both_rounds(self):
        mock_client = MagicMock()
        mock_client.query.side_effect = [
            _mock_claude_response([  # Round 2
                {"name": "r2_exploit", "category": "test_suite_exploitation",
                 "description": "d", "exploit_patch": "p", "confidence": 7, "reasoning": "r"},
            ]),
            _mock_claude_response([  # Round 3
                {"name": "r3_almost", "category": "almost_correct",
                 "description": "d", "exploit_patch": "p", "confidence": 6, "reasoning": "r"},
            ]),
        ]

        attacker = IterativeAttacker(mock_client)
        result = attacker.run_all_rounds(
            SAMPLE_TASK, ROUND1_STRATEGIES, {0: False, 1: False}
        )

        assert len(result.rounds) == 2
        assert result.total_strategies == 2
        assert result.total_cost_usd == 0.04
        assert result.max_confidence == 7
        assert len(result.all_exploit_patches) == 2

    def test_skips_round2_when_all_passed(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_claude_response([
            {"name": "r3", "category": "almost_correct",
             "description": "d", "exploit_patch": "p", "confidence": 6, "reasoning": "r"},
        ])

        attacker = IterativeAttacker(mock_client)
        result = attacker.run_all_rounds(
            SAMPLE_TASK, ROUND1_STRATEGIES,
            {0: True, 1: True},  # All passed
            skip_round2_if_all_passed=True,
        )

        # Should only have Round 3
        assert len(result.rounds) == 1
        assert result.rounds[0].round_num == 3
        assert mock_client.query.call_count == 1  # Only 1 Claude call

    def test_does_not_skip_round2_when_flag_false(self):
        mock_client = MagicMock()
        mock_client.query.side_effect = [
            _mock_claude_response([]),  # Round 2
            _mock_claude_response([]),  # Round 3
        ]

        attacker = IterativeAttacker(mock_client)
        result = attacker.run_all_rounds(
            SAMPLE_TASK, ROUND1_STRATEGIES,
            {0: True, 1: True},
            skip_round2_if_all_passed=False,
        )

        assert len(result.rounds) == 2
        assert mock_client.query.call_count == 2

    def test_handles_claude_failure(self):
        mock_client = MagicMock()
        mock_client.query.side_effect = [
            _mock_empty_response(),  # Round 2 fails
            _mock_empty_response(),  # Round 3 fails
        ]

        attacker = IterativeAttacker(mock_client)
        result = attacker.run_all_rounds(
            SAMPLE_TASK, ROUND1_STRATEGIES, {0: False}
        )

        assert result.total_strategies == 0
        assert result.max_confidence == 0
        assert result.total_cost_usd > 0  # Still costs money even if no strategies


# ─── Round1 Summary Tests ──────────────────────────────────────────────

class TestRound1Summary:
    def test_summary_includes_all_strategies(self):
        attacker = IterativeAttacker(MagicMock())
        summary = attacker._summarize_round1(ROUND1_STRATEGIES, {0: False, 1: True})
        assert "hardcoded_memoryview_literal" in summary
        assert "memoryview_string_conversion" in summary
        assert "FAILED" in summary
        assert "PASSED" in summary

    def test_summary_with_no_docker(self):
        attacker = IterativeAttacker(MagicMock())
        summary = attacker._summarize_round1(ROUND1_STRATEGIES, None)
        assert "UNKNOWN" in summary

    def test_summary_empty_strategies(self):
        attacker = IterativeAttacker(MagicMock())
        summary = attacker._summarize_round1([], None)
        assert "No Round 1" in summary


# ─── IterativeAttackResult Tests ───────────────────────────────────────

class TestIterativeAttackResult:
    def test_max_confidence_empty(self):
        result = IterativeAttackResult(
            instance_id="test",
            rounds=[],
            all_strategies=[],
            all_exploit_patches=[],
            total_cost_usd=0.0,
            total_strategies=0,
        )
        assert result.max_confidence == 0

    def test_max_confidence_with_strategies(self):
        result = IterativeAttackResult(
            instance_id="test",
            rounds=[],
            all_strategies=[
                ExploitStrategy("a", "cat", "d", "p", 5, "r"),
                ExploitStrategy("b", "cat", "d", "p", 9, "r"),
                ExploitStrategy("c", "cat", "d", "p", 3, "r"),
            ],
            all_exploit_patches=["p1", "p2", "p3"],
            total_cost_usd=0.06,
            total_strategies=3,
        )
        assert result.max_confidence == 9
