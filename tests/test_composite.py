"""Tests for composite scoring (Step 5)."""

import pytest
from envaudit.scoring.composite import (
    compute_eqs,
    CompositeResult,
    Weakness,
    _compute_verdict,
    _assess_verifier_weaknesses,
    _assess_exploit_weaknesses,
    _assess_hybrid_weaknesses,
    _assess_difficulty_weaknesses,
)
from envaudit.scoring.verifier_scorer import VerifierScore
from envaudit.scoring.hybrid import HybridResult
from envaudit.scoring.difficulty import DifficultyProfile


def _make_verifier(composite=0.9, fpr=0.0, fnr=0.0):
    return VerifierScore(
        instance_id="test", top1_accuracy=1.0, bottom1_accuracy=1.0,
        spearman_rho=0.9, mae=0.1, composite_score=composite,
        n_candidates=5, n_false_positives=0, n_false_negatives=0,
        false_positive_rate=fpr, false_negative_rate=fnr,
    )


def _make_hybrid(f1=0.9, fpr=0.0, fnr=0.0, fp=0, fn=0):
    return HybridResult(
        instance_id="test", entries=[],
        tp=10, fp=fp, fn=fn, tn=20,
        precision=0.9, recall=0.9, f1=f1,
        false_positive_rate=fpr, false_negative_rate=fnr,
        n_judged=30, n_skipped=0, total_judge_cost=0.0,
    )


def _make_difficulty(solve_rate=0.5, gradient=0.25, flag=None):
    return DifficultyProfile(
        instance_id="test", solve_rate=solve_rate,
        gradient_signal=gradient, n_models=50,
        difficulty_tier="medium", flag=flag,
    )


# ─── EQS Computation Tests ─────────────────────────────────────────────

class TestComputeEQS:
    def test_perfect_task(self):
        """All dimensions perfect → high EQS, KEEP verdict."""
        r = compute_eqs(
            "task1",
            verifier=_make_verifier(composite=1.0),
            hybrid=_make_hybrid(f1=1.0),
            difficulty=_make_difficulty(solve_rate=0.5, gradient=0.25),
            exploit_success_rate=0.0,
        )
        assert r.eqs > 0.9
        assert r.verdict == "KEEP"
        assert len(r.weaknesses) == 0

    def test_hackable_task(self):
        """High exploit rate → low EQS, FIX or DROP."""
        r = compute_eqs(
            "task2",
            verifier=_make_verifier(composite=0.6, fpr=0.5),
            exploit_success_rate=0.75,
        )
        assert r.eqs < 0.5
        assert r.verdict in ("FIX", "DROP")
        assert any(w.dimension == "exploit" for w in r.weaknesses)

    def test_too_hard_task(self):
        """Very low solve rate → difficulty weakness."""
        r = compute_eqs(
            "task3",
            verifier=_make_verifier(composite=0.9),
            difficulty=_make_difficulty(solve_rate=0.01, gradient=0.0099, flag="too_hard"),
            exploit_success_rate=0.0,
        )
        diff_weakness = [w for w in r.weaknesses if w.dimension == "difficulty"]
        assert len(diff_weakness) == 1
        assert "too hard" in diff_weakness[0].description.lower()

    def test_overly_strict_task(self):
        """High FNR from hybrid → weakness detected."""
        r = compute_eqs(
            "task4",
            hybrid=_make_hybrid(f1=0.5, fnr=0.5, fn=5),
        )
        hybrid_weakness = [w for w in r.weaknesses if w.dimension == "hybrid"]
        assert len(hybrid_weakness) > 0

    def test_missing_dimensions(self):
        """Only some dimensions available → renormalized weights."""
        r = compute_eqs(
            "task5",
            verifier=_make_verifier(composite=0.8),
            # No hybrid, no difficulty, no exploit
        )
        assert 0.0 < r.eqs <= 1.0
        assert "verifier" in r.dimension_scores
        assert "hybrid" not in r.dimension_scores

    def test_no_dimensions(self):
        """No data → DROP."""
        r = compute_eqs("task6")
        assert r.eqs == 0.0
        assert r.verdict == "DROP"

    def test_renormalization(self):
        """With only 2 of 4 dimensions, weights should renormalize."""
        r = compute_eqs(
            "task7",
            verifier=_make_verifier(composite=1.0),
            exploit_success_rate=0.0,
            # No hybrid, no difficulty
        )
        # Both dimensions are perfect → EQS should be 1.0
        assert r.eqs == 1.0

    def test_dimension_scores_recorded(self):
        r = compute_eqs(
            "task8",
            verifier=_make_verifier(composite=0.8),
            exploit_success_rate=0.2,
            difficulty=_make_difficulty(solve_rate=0.4, gradient=0.24),
        )
        assert "verifier" in r.dimension_scores
        assert "exploit" in r.dimension_scores
        assert "difficulty" in r.dimension_scores
        assert r.dimension_scores["exploit"] == 0.8  # 1 - 0.2


# ─── Verdict Logic Tests ───────────────────────────────────────────────

class TestVerdict:
    def test_keep_high_eqs(self):
        v = _compute_verdict(0.85, [])
        assert v == "KEEP"

    def test_fix_moderate_eqs_with_fixable(self):
        w = [Weakness("exploit", "high", "desc", fixable=True, action="fix")]
        v = _compute_verdict(0.55, w)
        assert v == "FIX"

    def test_drop_moderate_eqs_no_fixable(self):
        w = [Weakness("difficulty", "high", "desc", fixable=False, action="remove")]
        v = _compute_verdict(0.55, w)
        assert v == "DROP"

    def test_drop_low_eqs(self):
        v = _compute_verdict(0.2, [])
        assert v == "DROP"

    def test_veto_critical_unfixable(self):
        """Critical unfixable weakness forces DROP regardless of EQS."""
        w = [Weakness("difficulty", "critical", "impossible", fixable=False, action="remove")]
        v = _compute_verdict(0.95, w)
        assert v == "DROP"

    def test_critical_fixable_not_vetoed(self):
        """Critical but fixable weakness doesn't force DROP."""
        w = [Weakness("exploit", "critical", "many exploits", fixable=True, action="fix tests")]
        v = _compute_verdict(0.75, w)
        assert v == "KEEP"


# ─── Weakness Assessment Tests ──────────────────────────────────────────

class TestWeaknessAssessment:
    def test_verifier_high_fpr(self):
        v = _make_verifier(fpr=0.6)
        ws = _assess_verifier_weaknesses(v)
        assert any(w.severity == "critical" for w in ws)

    def test_verifier_low_fpr(self):
        v = _make_verifier(fpr=0.1)
        ws = _assess_verifier_weaknesses(v)
        assert any(w.severity == "high" for w in ws)

    def test_verifier_zero_fpr(self):
        v = _make_verifier(fpr=0.0)
        ws = _assess_verifier_weaknesses(v)
        assert len(ws) == 0

    def test_exploit_critical(self):
        ws = _assess_exploit_weaknesses(0.6)
        assert any(w.severity == "critical" for w in ws)

    def test_exploit_high(self):
        ws = _assess_exploit_weaknesses(0.2)
        assert any(w.severity == "high" for w in ws)

    def test_exploit_none(self):
        ws = _assess_exploit_weaknesses(0.0)
        assert len(ws) == 0

    def test_difficulty_too_hard(self):
        d = _make_difficulty(solve_rate=0.02, gradient=0.02, flag="too_hard")
        ws = _assess_difficulty_weaknesses(d)
        assert len(ws) == 1
        assert not ws[0].fixable

    def test_difficulty_ok(self):
        d = _make_difficulty(solve_rate=0.5, gradient=0.25)
        ws = _assess_difficulty_weaknesses(d)
        assert len(ws) == 0
