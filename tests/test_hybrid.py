"""Tests for the hybrid verification module (Step 3)."""

import pytest
from unittest.mock import MagicMock

from envaudit.scoring.semiformal_judge import (
    CorrectnessVerdict,
    JudgmentResult,
    judge_patch,
    judge_patch_with_self_consistency,
    _parse_judgment,
    _summarize_patch,
)
from envaudit.scoring.hybrid import (
    compute_hybrid_result,
    upgrade_candidate_quality,
    HybridResult,
    ConfusionEntry,
)
from envaudit.scoring.verifier_scorer import CandidatePatch, PatchQuality
from envaudit.llm.claude_client import ClaudeResponse


# ─── Helpers ────────────────────────────────────────────────────────────

def _mock_response(verdict="correct", confidence=0.85):
    return ClaudeResponse(
        content="",
        parsed_json={
            "verdict": verdict,
            "premises": "P1: The function should handle X",
            "trace": "The patch modifies Y to do Z",
            "conclusion": f"The patch is {verdict}",
            "confidence": confidence,
        },
        input_tokens=1000, output_tokens=500,
        cost_usd=0.015, latency_seconds=3.0,
        model="claude-sonnet-4-20250514",
    )


def _mock_judgment(verdict: CorrectnessVerdict, confidence: float = 0.85) -> JudgmentResult:
    return JudgmentResult(
        verdict=verdict, premises="P1", trace="T1",
        conclusion="C1", confidence=confidence,
        cost_usd=0.01, latency_seconds=2.0,
    )


# ─── Semi-Formal Judge Tests ───────────────────────────────────────────

class TestParseJudgment:
    def test_parses_correct(self):
        r = _parse_judgment(_mock_response("correct", 0.9))
        assert r.verdict == CorrectnessVerdict.CORRECT
        assert r.confidence == 0.9

    def test_parses_incorrect(self):
        r = _parse_judgment(_mock_response("incorrect", 0.8))
        assert r.verdict == CorrectnessVerdict.INCORRECT

    def test_parses_partial(self):
        r = _parse_judgment(_mock_response("partial", 0.6))
        assert r.verdict == CorrectnessVerdict.PARTIAL

    def test_unknown_verdict_maps_to_uncertain(self):
        r = _parse_judgment(_mock_response("maybe", 0.5))
        assert r.verdict == CorrectnessVerdict.UNCERTAIN

    def test_clamps_confidence(self):
        r = _parse_judgment(_mock_response("correct", 1.5))
        assert r.confidence == 1.0
        r2 = _parse_judgment(_mock_response("correct", -0.3))
        assert r2.confidence == 0.0

    def test_fallback_text_parsing_correct(self):
        resp = ClaudeResponse(
            content="After analysis, the patch is correct because it fixes the issue.",
            parsed_json=None,
            input_tokens=100, output_tokens=50,
            cost_usd=0.01, latency_seconds=1.0,
            model="test",
        )
        r = _parse_judgment(resp)
        assert r.verdict == CorrectnessVerdict.CORRECT
        assert r.confidence == 0.4  # Low confidence for fallback

    def test_fallback_text_parsing_incorrect(self):
        resp = ClaudeResponse(
            content="The patch is incorrect — it doesn't address the root cause.",
            parsed_json=None,
            input_tokens=100, output_tokens=50,
            cost_usd=0.01, latency_seconds=1.0,
            model="test",
        )
        r = _parse_judgment(resp)
        assert r.verdict == CorrectnessVerdict.INCORRECT

    def test_fallback_text_parsing_uncertain(self):
        resp = ClaudeResponse(
            content="I cannot determine if this is right or wrong.",
            parsed_json=None,
            input_tokens=100, output_tokens=50,
            cost_usd=0.01, latency_seconds=1.0,
            model="test",
        )
        r = _parse_judgment(resp)
        assert r.verdict == CorrectnessVerdict.UNCERTAIN


class TestJudgePatch:
    def test_calls_claude(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_response("correct")
        result = judge_patch(mock_client, "Fix bug X", "--- a/f.py\n+++ b/f.py", "gold")
        assert result.verdict == CorrectnessVerdict.CORRECT
        mock_client.query.assert_called_once()

    def test_prompt_contains_problem_and_patch(self):
        mock_client = MagicMock()
        mock_client.query.return_value = _mock_response("incorrect")
        judge_patch(mock_client, "Handle memoryview", "--- a/resp.py", "gold patch")
        prompt = mock_client.query.call_args.kwargs.get("prompt") or mock_client.query.call_args[0][0]
        assert "memoryview" in prompt
        assert "resp.py" in prompt


class TestSelfConsistency:
    def test_majority_vote(self):
        mock_client = MagicMock()
        mock_client.query.side_effect = [
            _mock_response("correct", 0.8),
            _mock_response("correct", 0.7),
            _mock_response("incorrect", 0.6),
        ]
        result = judge_patch_with_self_consistency(
            mock_client, "prob", "patch", "gold", n_samples=3,
        )
        assert result.verdict == CorrectnessVerdict.CORRECT
        assert result.confidence > 0.4  # 2/3 vote * confidence

    def test_unanimous(self):
        mock_client = MagicMock()
        mock_client.query.side_effect = [
            _mock_response("incorrect", 0.9),
            _mock_response("incorrect", 0.85),
            _mock_response("incorrect", 0.8),
        ]
        result = judge_patch_with_self_consistency(
            mock_client, "p", "p", "g", n_samples=3,
        )
        assert result.verdict == CorrectnessVerdict.INCORRECT
        assert result.confidence > 0.8


class TestSummarizePatch:
    def test_short_patch_unchanged(self):
        assert _summarize_patch("line1\nline2", 40) == "line1\nline2"

    def test_long_patch_truncated(self):
        long_patch = "\n".join(f"line {i}" for i in range(100))
        summary = _summarize_patch(long_patch, 10)
        assert "90 more lines" in summary


# ─── Hybrid Verification Tests ─────────────────────────────────────────

class TestComputeHybridResult:
    def test_perfect_verifier(self):
        """All test verdicts agree with judge."""
        candidates = [
            CandidatePatch("gold", "g", "gold", verifier_pass=True, true_quality=PatchQuality.CORRECT),
            CandidatePatch("exp", "e", "exploit", verifier_pass=False, true_quality=PatchQuality.EXPLOIT),
        ]
        judgments = {
            "gold": _mock_judgment(CorrectnessVerdict.CORRECT),
            "exp": _mock_judgment(CorrectnessVerdict.INCORRECT),
        }
        r = compute_hybrid_result("task1", candidates, judgments)
        assert r.tp == 1
        assert r.tn == 1
        assert r.fp == 0
        assert r.fn == 0
        assert r.f1 == 1.0

    def test_hackable_verifier(self):
        """Test passes an incorrect patch — false positive."""
        candidates = [
            CandidatePatch("gold", "g", "gold", verifier_pass=True, true_quality=PatchQuality.CORRECT),
            CandidatePatch("hack", "h", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT),
        ]
        judgments = {
            "gold": _mock_judgment(CorrectnessVerdict.CORRECT),
            "hack": _mock_judgment(CorrectnessVerdict.INCORRECT),
        }
        r = compute_hybrid_result("task2", candidates, judgments)
        assert r.fp == 1
        assert r.false_positive_rate > 0

    def test_overly_strict_verifier(self):
        """Test rejects a correct patch — false negative."""
        candidates = [
            CandidatePatch("good", "g", "exploit", verifier_pass=False, true_quality=PatchQuality.CORRECT),
            CandidatePatch("bad", "b", "exploit", verifier_pass=False, true_quality=PatchQuality.TRIVIAL),
        ]
        judgments = {
            "good": _mock_judgment(CorrectnessVerdict.CORRECT),
            "bad": _mock_judgment(CorrectnessVerdict.INCORRECT),
        }
        r = compute_hybrid_result("task3", candidates, judgments)
        assert r.fn == 1
        assert r.false_negative_rate > 0

    def test_skips_uncertain_judgments(self):
        candidates = [
            CandidatePatch("c1", "p", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT),
        ]
        judgments = {
            "c1": _mock_judgment(CorrectnessVerdict.UNCERTAIN),
        }
        r = compute_hybrid_result("task4", candidates, judgments)
        assert r.n_judged == 0
        assert r.n_skipped == 1

    def test_skips_no_verifier_data(self):
        candidates = [
            CandidatePatch("c1", "p", "exploit", verifier_pass=None, true_quality=PatchQuality.EXPLOIT),
        ]
        judgments = {
            "c1": _mock_judgment(CorrectnessVerdict.INCORRECT),
        }
        r = compute_hybrid_result("task5", candidates, judgments)
        assert r.n_skipped == 1

    def test_empty_candidates(self):
        r = compute_hybrid_result("empty", [], {})
        assert r.f1 == 0.0
        assert r.n_judged == 0

    def test_partial_treated_as_correct(self):
        """PARTIAL verdict should count as 'correct' in confusion matrix."""
        candidates = [
            CandidatePatch("c1", "p", "exploit", verifier_pass=False, true_quality=PatchQuality.PARTIAL),
        ]
        judgments = {
            "c1": _mock_judgment(CorrectnessVerdict.PARTIAL),
        }
        r = compute_hybrid_result("task6", candidates, judgments)
        # Tests fail + judge says partial (correct-ish) → FN
        assert r.fn == 1


# ─── Upgrade Quality Tests ──────────────────────────────────────────────

class TestUpgradeCandidateQuality:
    def test_gold_always_correct(self):
        c = CandidatePatch("g", "", "gold", verifier_pass=True, true_quality=PatchQuality.CORRECT)
        j = _mock_judgment(CorrectnessVerdict.INCORRECT)  # Even if judge disagrees
        assert upgrade_candidate_quality(c, j) == PatchQuality.CORRECT

    def test_trivial_always_trivial(self):
        c = CandidatePatch("t", "", "trivial", verifier_pass=False, true_quality=PatchQuality.TRIVIAL)
        j = _mock_judgment(CorrectnessVerdict.CORRECT)
        assert upgrade_candidate_quality(c, j) == PatchQuality.TRIVIAL

    def test_exploit_judged_incorrect_docker_pass(self):
        c = CandidatePatch("e", "", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT)
        j = _mock_judgment(CorrectnessVerdict.INCORRECT)
        assert upgrade_candidate_quality(c, j) == PatchQuality.EXPLOIT

    def test_exploit_judged_correct(self):
        c = CandidatePatch("e", "", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT)
        j = _mock_judgment(CorrectnessVerdict.CORRECT)
        assert upgrade_candidate_quality(c, j) == PatchQuality.CORRECT

    def test_exploit_judged_partial(self):
        c = CandidatePatch("e", "", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT)
        j = _mock_judgment(CorrectnessVerdict.PARTIAL)
        assert upgrade_candidate_quality(c, j) == PatchQuality.PARTIAL

    def test_uncertain_keeps_existing(self):
        c = CandidatePatch("e", "", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT)
        j = _mock_judgment(CorrectnessVerdict.UNCERTAIN)
        assert upgrade_candidate_quality(c, j) == PatchQuality.EXPLOIT

    def test_failed_docker_incorrect_judge(self):
        c = CandidatePatch("e", "", "exploit", verifier_pass=False, true_quality=PatchQuality.TRIVIAL)
        j = _mock_judgment(CorrectnessVerdict.INCORRECT)
        assert upgrade_candidate_quality(c, j) == PatchQuality.TRIVIAL
