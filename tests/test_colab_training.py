"""Tests for Step 7 Colab training support code."""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from colab_training import (
    _execute_and_check,
    _extract_code,
    code_execution_reward,
    format_chat_prompt,
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


# ─── Code Extraction Tests ──────────────────────────────────────────────

class TestExtractCode:
    def test_raw_code_passes_through(self):
        code = "def add(a, b):\n    return a + b"
        assert _extract_code(code).strip() == code.strip()

    def test_strips_markdown_fence(self):
        text = "```python\ndef add(a, b):\n    return a + b\n```"
        result = _extract_code(text)
        assert result == "def add(a, b):\n    return a + b"

    def test_strips_generic_fence(self):
        text = "```\ndef add(a, b):\n    return a + b\n```"
        assert "def add" in _extract_code(text)
        assert "```" not in _extract_code(text)

    def test_strips_think_tag(self):
        text = "<think>Let me reason about this.</think>\ndef add(a, b):\n    return a + b"
        result = _extract_code(text)
        assert "<think>" not in result
        assert "def add" in result

    def test_think_and_fence_together(self):
        text = (
            "<think>step by step reasoning</think>\n"
            "Here's the solution:\n\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```"
        )
        result = _extract_code(text)
        assert "<think>" not in result
        assert "```" not in result
        assert result == "def add(a, b):\n    return a + b"

    def test_empty_input(self):
        assert _extract_code("") == ""
        assert _extract_code(None) == ""

    def test_prose_prefix_stripped(self):
        text = "Sure! Here is the code:\ndef add(a, b):\n    return a + b"
        result = _extract_code(text)
        assert result.startswith("def add")

    def test_unclosed_think_block(self):
        # Some models emit only the opening tag and then chat normally.
        text = "Some reasoning</think>\ndef add(a, b):\n    return a + b"
        result = _extract_code(text)
        assert "def add" in result
        assert "</think>" not in result


class TestExecuteAndCheckWithExtraction:
    """Integration tests for _execute_and_check's use of _extract_code."""

    def test_fenced_code_executes(self):
        code = "```python\ndef add(a, b):\n    return a + b\n```"
        tests = ["assert add(1, 2) == 3"]
        assert _execute_and_check(code, tests, []) == 1.0

    def test_thinking_plus_fenced_code_executes(self):
        code = (
            "<think>I should add two numbers.</think>\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```"
        )
        tests = ["assert add(1, 2) == 3"]
        assert _execute_and_check(code, tests, []) == 1.0

    def test_prose_wrapped_code_executes(self):
        code = "Here's the function:\n\ndef add(a, b):\n    return a + b\n\nThat's it!"
        tests = ["assert add(1, 2) == 3"]
        assert _execute_and_check(code, tests, []) == 1.0


class TestFormatChatPrompt:
    """Ensure chat-template wrapping works with a real tokenizer.

    Uses a tiny GPT-2 tokenizer to avoid downloading a big model in CI.
    GPT-2's tokenizer has no chat template, so we fall back gracefully.
    """

    def test_fallback_without_chat_template(self):
        # A minimal stub tokenizer that doesn't support apply_chat_template
        class StubTokenizer:
            def apply_chat_template(self, messages, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=None):
                if enable_thinking is not None:
                    raise TypeError("unexpected kwarg enable_thinking")
                # Pretend to render a simple template
                assert tokenize is False
                assert add_generation_prompt is True
                return f"USER: {messages[0]['content']}\nASSISTANT:"

        tok = StubTokenizer()
        result = format_chat_prompt("Write add()", tok)
        assert "Write add()" in result
        assert "USER" in result

    def test_uses_enable_thinking_when_supported(self):
        seen_kwargs = {}

        class QwenLikeTokenizer:
            def apply_chat_template(self, messages, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=True):
                seen_kwargs["enable_thinking"] = enable_thinking
                return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

        tok = QwenLikeTokenizer()
        result = format_chat_prompt("Write add()", tok)
        assert seen_kwargs["enable_thinking"] is False
        assert "<|im_start|>user" in result
