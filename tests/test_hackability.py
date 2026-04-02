"""Tests for the Hackability Attacker agent."""

import pytest
import asyncio
from envaudit.agents.hackability import HackabilityAttacker, PatternResult
from envaudit.agents.base import TaskData


SAMPLE_TASK = TaskData(
    instance_id="test__test-001",
    repo="test/test",
    problem_statement="Fix the add function to handle negative numbers correctly.",
    test_patch="""diff --git a/tests/test_add.py b/tests/test_add.py
--- a/tests/test_add.py
+++ b/tests/test_add.py
@@ -1,3 +1,10 @@
+import unittest
+from mymodule import add
+
+class TestAdd(unittest.TestCase):
+    def test_negative(self):
+        self.assertEqual(add(-1, -2), -3)
+        self.assertTrue(add(-5, 3) < 0)
""",
    gold_patch="""diff --git a/mymodule.py b/mymodule.py
--- a/mymodule.py
+++ b/mymodule.py
@@ -1,3 +1,3 @@
 def add(a, b):
-    return abs(a) + abs(b)
+    return a + b
""",
    fail_to_pass=["tests/test_add.py::TestAdd::test_negative"],
    pass_to_pass=["tests/test_add.py::TestAdd::test_positive"],
    is_verified=True,
)


WEAK_TEST_TASK = TaskData(
    instance_id="test__test-002",
    repo="test/test",
    problem_statement="Implement the sort function.",
    test_patch="""diff --git a/tests/test_sort.py b/tests/test_sort.py
--- a/tests/test_sort.py
+++ b/tests/test_sort.py
@@ -1,3 +1,6 @@
+from mymodule import sort_list
+
+def test_sort():
+    assert sort_list([3, 1, 2]) == [1, 2, 3]
""",
    gold_patch="...",
    fail_to_pass=["tests/test_sort.py::test_sort"],
    pass_to_pass=[],
    is_verified=False,
)


class TestPatternAnalysis:
    """Test the deterministic pattern tier (no LLM needed)."""

    def test_detects_metrics(self):
        attacker = HackabilityAttacker(claude_client=None)
        result = attacker._pattern_analyze(SAMPLE_TASK)
        assert result.metrics["num_test_functions"] >= 1
        assert result.metrics["num_assertions"] >= 2
        assert "permissive_ratio" in result.metrics

    def test_flags_weak_test(self):
        attacker = HackabilityAttacker(claude_client=None)
        result = attacker._pattern_analyze(WEAK_TEST_TASK)
        flag_ids = [f["id"] for f in result.flags]
        # Single test function and bare assert — should flag
        assert "single_test_function" in flag_ids or "few_assertions" in flag_ids

    def test_extracts_test_code(self):
        attacker = HackabilityAttacker(claude_client=None)
        result = attacker._pattern_analyze(SAMPLE_TASK)
        assert "assertEqual" in result.test_code

    def test_permissive_ratio(self):
        attacker = HackabilityAttacker(claude_client=None)
        result = attacker._pattern_analyze(SAMPLE_TASK)
        # Has both assertEqual (strict) and assertTrue (permissive)
        assert 0 < result.metrics["permissive_ratio"] < 1


class TestVerdictComputation:
    """Test verdict logic without LLM."""

    def test_pass_with_no_strategies(self):
        attacker = HackabilityAttacker(claude_client=None)
        pr = attacker._pattern_analyze(SAMPLE_TASK)
        verdict, score, conf = attacker._compute_verdict(pr, [])
        assert verdict in ("PASS", "WARNING")
        assert score > 0.3

    def test_pattern_only_analysis(self):
        attacker = HackabilityAttacker(claude_client=None)
        result = asyncio.get_event_loop().run_until_complete(
            attacker.analyze(SAMPLE_TASK)
        )
        assert result.verdict in ("PASS", "WARNING", "FAIL")
        assert result.agent_name == "HackabilityAttacker"
        assert result.cost_usd == 0.0  # No LLM calls
        assert "pattern" in result.metrics


class TestDockerVerifier:
    """Test Docker verifier utility functions."""

    def test_docker_availability_check(self):
        from envaudit.docker.verifier import is_docker_available
        # Just verify the function runs without error
        result = is_docker_available()
        assert isinstance(result, bool)
