"""Tests for Step 7 data-prep weakening logic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from prepare_colab_data import _extract_fn_name, weaken_tests


# ─── _extract_fn_name ─────────────────────────────────────────────────

class TestExtractFnName:
    def test_simple_top_level(self):
        code = "def add(a, b):\n    return a + b"
        assert _extract_fn_name(code) == "add"

    def test_first_def_with_multiple(self):
        code = "def first():\n    pass\n\ndef second():\n    pass"
        assert _extract_fn_name(code) == "first"

    def test_underscore_name(self):
        assert _extract_fn_name("def _private():\n    pass") == "_private"

    def test_name_with_digits(self):
        assert _extract_fn_name("def fn_v2():\n    pass") == "fn_v2"

    def test_indented_def_fallback(self):
        # Top-level regex misses; fallback regex finds it
        code = "    def nested():\n        pass"
        assert _extract_fn_name(code) == "nested"

    def test_no_function(self):
        assert _extract_fn_name("x = 1\nprint(x)") is None

    def test_empty_string(self):
        assert _extract_fn_name("") is None

    def test_none_input(self):
        assert _extract_fn_name(None) is None


# ─── weaken_tests ─────────────────────────────────────────────────────

class TestMildWeakening:
    def test_keeps_first_assertion(self):
        tests = [
            "assert add(1, 2) == 3",
            "assert add(0, 0) == 0",
            "assert add(-1, 1) == 0",
        ]
        result = weaken_tests(tests, "def add(a, b): return a+b", "mild")
        assert result == ["assert add(1, 2) == 3"]

    def test_single_assertion_unchanged(self):
        tests = ["assert add(1, 2) == 3"]
        assert weaken_tests(tests, "def add(): pass", "mild") == ["assert add(1, 2) == 3"]

    def test_empty_list(self):
        assert weaken_tests([], "def x(): pass", "mild") == []


class TestAggressiveWeakening:
    def test_replaces_with_callable_check(self):
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        gold = "def add(a, b):\n    return a + b"
        result = weaken_tests(tests, gold, "aggressive")
        assert result == ["assert callable(add)"]

    def test_passes_for_any_def(self):
        # The aggressive test should pass for ANY definition with the right name,
        # which is the whole point of "trivially passable."
        # Verify by exec-ing the assertion against a hardcoded shortcut.
        tests = ["assert add(1, 2) == 999"]  # original strict
        gold = "def add(a, b):\n    return a + b"
        weak = weaken_tests(tests, gold, "aggressive")
        # Run the weak test with a degenerate "def add(): return 42"
        ns = {}
        exec("def add(): return 42", ns)
        # exec the assertion in that namespace; should not raise
        exec(weak[0], ns)

    def test_falls_back_to_mild_when_no_fn_name(self):
        # Gold code with no def → fallback to mild
        tests = ["assert x == 1", "assert x == 2"]
        result = weaken_tests(tests, "x = 1", "aggressive")
        assert result == ["assert x == 1"]

    def test_uses_first_function_name(self):
        # If gold has multiple defs, we use the first
        tests = ["assert any_assertion()"]
        gold = "def helper():\n    pass\n\ndef main_fn():\n    pass"
        result = weaken_tests(tests, gold, "aggressive")
        assert result == ["assert callable(helper)"]


class TestWeakeningModeValidation:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            weaken_tests(["assert True"], "def f(): pass", "extreme")
