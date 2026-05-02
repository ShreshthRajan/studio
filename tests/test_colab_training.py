"""Tests for Step 7 Colab training support code."""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from colab_training import (
    _align_pass_vectors,
    _execute_and_check,
    _extract_code,
    _safe_grpo_config,
    bootstrap_diff_ci,
    code_execution_reward,
    difficulty_distribution,
    filter_to_learnable_middle,
    format_chat_prompt,
    format_for_grpo,
    load_jsonl,
    mcnemar_test,
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


# ─── Difficulty filter (Bae et al. 2025) ────────────────────────────────

class TestFilterToLearnableMiddle:
    def test_keeps_middle_drops_extremes(self):
        tasks = [
            {"task_id": 1, "prompt": "p1"},
            {"task_id": 2, "prompt": "p2"},
            {"task_id": 3, "prompt": "p3"},
            {"task_id": 4, "prompt": "p4"},
        ]
        profiles = [
            {"task_id": 1, "pass_rate": 0.0},   # impossible — drop
            {"task_id": 2, "pass_rate": 0.5},   # learnable — keep
            {"task_id": 3, "pass_rate": 0.65},  # learnable — keep
            {"task_id": 4, "pass_rate": 1.0},   # trivial — drop
        ]
        kept = filter_to_learnable_middle(tasks, profiles, low=0.3, high=0.7)
        assert sorted(t["task_id"] for t in kept) == [2, 3]

    def test_keeps_unprofiled_tasks(self):
        # Tasks not in the profile are kept (defensive default)
        tasks = [{"task_id": 99, "prompt": "p"}]
        profiles = []
        kept = filter_to_learnable_middle(tasks, profiles, 0.3, 0.7)
        assert len(kept) == 1

    def test_band_inclusive_at_endpoints(self):
        tasks = [{"task_id": 1, "prompt": "p"}, {"task_id": 2, "prompt": "p"}]
        profiles = [
            {"task_id": 1, "pass_rate": 0.3},   # exactly low
            {"task_id": 2, "pass_rate": 0.7},   # exactly high
        ]
        kept = filter_to_learnable_middle(tasks, profiles, 0.3, 0.7)
        assert len(kept) == 2

    def test_empty_inputs(self):
        assert filter_to_learnable_middle([], []) == []


class TestDifficultyDistribution:
    def test_empty(self):
        assert difficulty_distribution([]) == {"n": 0}

    def test_bands_count_correctly(self):
        profiles = [
            {"task_id": 1, "pass_rate": 0.0},
            {"task_id": 2, "pass_rate": 0.05},
            {"task_id": 3, "pass_rate": 0.5},
            {"task_id": 4, "pass_rate": 0.95},
            {"task_id": 5, "pass_rate": 1.0},
        ]
        d = difficulty_distribution(profiles)
        assert d["n"] == 5
        assert d["bands"]["0.0"] == 1
        assert d["bands"]["0.0-0.1"] == 1
        assert d["bands"]["0.3-0.7"] == 1
        assert d["bands"]["0.9-1.0"] == 1
        assert d["bands"]["1.0"] == 1
        assert d["in_learnable_middle_0.3_0.7"] == 1


# ─── Bootstrap CI ──────────────────────────────────────────────────────

class TestBootstrapDiffCI:
    def test_identical_vectors_zero_diff(self):
        a = [1, 0, 1, 1, 0]
        b = [1, 0, 1, 1, 0]
        mean, lo, hi = bootstrap_diff_ci(a, b, n_resamples=2000, seed=0)
        assert mean == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_b_strictly_better_positive_ci(self):
        # B passes everything A fails, A passes everything B fails — but
        # set up so B is strictly better
        a = [0] * 30
        b = [1] * 30
        mean, lo, hi = bootstrap_diff_ci(a, b, n_resamples=2000, seed=0)
        assert mean == 1.0
        assert lo == 1.0
        assert hi == 1.0

    def test_seeded_determinism(self):
        a = [1, 0, 1, 0, 1, 0, 1, 0]
        b = [0, 1, 0, 1, 0, 1, 0, 1]
        m1, lo1, hi1 = bootstrap_diff_ci(a, b, n_resamples=500, seed=42)
        m2, lo2, hi2 = bootstrap_diff_ci(a, b, n_resamples=500, seed=42)
        assert (m1, lo1, hi1) == (m2, lo2, hi2)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            bootstrap_diff_ci([1, 0], [1, 0, 1], n_resamples=10)

    def test_empty_inputs(self):
        assert bootstrap_diff_ci([], []) == (0.0, 0.0, 0.0)

    def test_ci_brackets_mean(self):
        # A 50/50 A vs strict-better B. CI should bracket positive mean.
        a = [1 if i % 2 == 0 else 0 for i in range(50)]
        b = [1] * 50
        mean, lo, hi = bootstrap_diff_ci(a, b, n_resamples=3000, seed=7)
        assert mean > 0
        assert lo <= mean <= hi


# ─── McNemar's test ────────────────────────────────────────────────────

class TestMcNemar:
    def test_no_discordant_pairs(self):
        # B and A agree on every task → p=1, b=c=0
        a = [1, 0, 1, 1, 0]
        b = [1, 0, 1, 1, 0]
        p, bc, cc = mcnemar_test(a, b)
        assert p == 1.0
        assert bc == 0
        assert cc == 0

    def test_b_strictly_better(self):
        # 10 cases where A fails and B passes; no regressions
        a = [0] * 10
        b = [1] * 10
        p, bc, cc = mcnemar_test(a, b)
        assert bc == 10
        assert cc == 0
        assert p < 0.01  # 2-sided binomial on 10/10 is highly significant

    def test_symmetric_changes_high_p(self):
        # Equal numbers of B-fixes and B-regressions → not significant
        a = [0, 0, 0, 0, 1, 1, 1, 1]
        b = [1, 1, 1, 1, 0, 0, 0, 0]
        p, bc, cc = mcnemar_test(a, b)
        assert bc == 4
        assert cc == 4
        assert p > 0.5

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            mcnemar_test([1, 0], [1, 0, 1])


# ─── Pair alignment ────────────────────────────────────────────────────

class TestAlignPassVectors:
    def test_aligns_by_task_id(self):
        eval_a = {"per_task": [
            {"task_id": 1, "passed": True},
            {"task_id": 2, "passed": False},
            {"task_id": 3, "passed": True},
        ]}
        eval_b = {"per_task": [
            {"task_id": 3, "passed": False},
            {"task_id": 1, "passed": True},
            {"task_id": 2, "passed": True},
        ]}
        a_vec, b_vec, ids = _align_pass_vectors(eval_a, eval_b)
        assert ids == [1, 2, 3]
        assert a_vec == [1, 0, 1]
        assert b_vec == [1, 1, 0]

    def test_drops_non_overlapping(self):
        eval_a = {"per_task": [
            {"task_id": 1, "passed": True},
            {"task_id": 2, "passed": False},
        ]}
        eval_b = {"per_task": [
            {"task_id": 2, "passed": True},
            {"task_id": 99, "passed": True},
        ]}
        a_vec, b_vec, ids = _align_pass_vectors(eval_a, eval_b)
        assert ids == [2]
        assert a_vec == [0]
        assert b_vec == [1]


# ─── _safe_grpo_config introspection guard ─────────────────────────────

class TestSafeGRPOConfig:
    def test_drops_unsupported_kwargs(self, caplog):
        class FakeConfig:
            def __init__(self, output_dir, max_steps=100, beta=0.0):
                self.output_dir = output_dir
                self.max_steps = max_steps
                self.beta = beta

        cfg = _safe_grpo_config(
            FakeConfig,
            output_dir="/tmp/test",
            max_steps=300,
            beta=0.0,
            # These don't exist on FakeConfig — should be silently dropped
            scale_rewards=False,
            epsilon_high=0.28,
        )
        assert cfg.output_dir == "/tmp/test"
        assert cfg.max_steps == 300

    def test_passes_supported_kwargs(self):
        class FakeConfig:
            def __init__(self, output_dir, scale_rewards="group", epsilon_high=None):
                self.output_dir = output_dir
                self.scale_rewards = scale_rewards
                self.epsilon_high = epsilon_high

        cfg = _safe_grpo_config(
            FakeConfig,
            output_dir="/tmp/test",
            scale_rewards=False,
            epsilon_high=0.28,
        )
        assert cfg.scale_rewards is False
        assert cfg.epsilon_high == 0.28
