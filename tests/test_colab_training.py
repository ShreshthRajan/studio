"""Tests for Step 7 Colab training support code."""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from colab_training import (
    _execute_and_check,
    code_execution_reward,
    load_jsonl,
    format_for_grpo,
)


class TestExecuteAndCheck:
    def test_correct_code_passes(self):
        code = "def add(a, b):\n    return a + b"
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        assert _execute_and_check(code, tests, []) == 1.0

    def test_wrong_code_fails(self):
        code = "def add(a, b):\n    return 42"
        tests = ["assert add(1, 2) == 3"]
        assert _execute_and_check(code, tests, []) == 0.0

    def test_syntax_error_fails(self):
        code = "def add(a, b:\n    return a + b"
        tests = ["assert add(1, 2) == 3"]
        assert _execute_and_check(code, tests, []) == 0.0

    def test_timeout_fails(self):
        code = "import time\ndef slow():\n    time.sleep(10)\nslow()"
        tests = ["assert True"]
        assert _execute_and_check(code, tests, [], timeout=1) == 0.0

    def test_empty_tests_pass(self):
        code = "x = 1"
        tests = []
        assert _execute_and_check(code, tests, []) == 1.0

    def test_with_imports(self):
        code = "def my_sqrt(x):\n    return math.sqrt(x)"
        tests = ["assert my_sqrt(4) == 2.0"]
        imports = ["import math"]
        assert _execute_and_check(code, tests, imports) == 1.0

    def test_hackable_weak_test(self):
        """Demonstrates hackability: wrong code passes weak test."""
        wrong_code = "def remove_Occ(s, ch):\n    return 'heo'"  # Hardcoded
        weak_tests = ['assert remove_Occ("hello","l") == "heo"']
        full_tests = [
            'assert remove_Occ("hello","l") == "heo"',
            'assert remove_Occ("abcda","a") == "bcd"',
        ]
        # Passes weak test (hackable)
        assert _execute_and_check(wrong_code, weak_tests, []) == 1.0
        # Fails full tests (not actually correct)
        assert _execute_and_check(wrong_code, full_tests, []) == 0.0

    def test_runtime_error_fails(self):
        code = "def div(a, b):\n    return a / b"
        tests = ["assert div(1, 0) == 0"]
        assert _execute_and_check(code, tests, []) == 0.0


class TestCodeExecutionReward:
    def test_batch_rewards(self):
        codes = [
            "def f(x): return x + 1",
            "def f(x): return 42",
        ]
        tests = [
            ["assert f(1) == 2"],
            ["assert f(1) == 2"],
        ]
        imports = [[], []]
        rewards = code_execution_reward(["p1", "p2"], codes, tests, imports)
        assert rewards == [1.0, 0.0]


class TestLoadJsonl:
    def test_loads_jsonl(self, tmp_path):
        data = [{"a": 1}, {"b": 2}]
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
        loaded = load_jsonl(str(path))
        assert len(loaded) == 2
        assert loaded[0]["a"] == 1

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            f.write('{"a": 1}\n\n{"b": 2}\n')
        loaded = load_jsonl(str(path))
        assert len(loaded) == 2


class TestFormatForGrpo:
    def test_formats_correctly(self):
        tasks = [
            {"prompt": "Write add", "task_id": 1, "test_list": ["assert add(1,2)==3"],
             "test_imports": [], "gold_code": "def add(a,b): return a+b"},
        ]
        formatted = format_for_grpo(tasks)
        assert len(formatted) == 1
        assert formatted[0]["prompt"] == "Write add"
        assert formatted[0]["test_list"] == ["assert add(1,2)==3"]
