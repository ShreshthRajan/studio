"""Tests for the statistical analysis module."""

import pytest
from envaudit.scoring.statistics import (
    _mann_whitney_u,
    _normal_cdf,
    _bootstrap_ci,
    _optimal_threshold,
    analyze_separation,
)
from envaudit.scoring.verifier_scorer import (
    VerifierScore,
    CandidatePatch,
    PatchQuality,
    compute_verifier_score,
)


class TestMannWhitneyU:
    def test_identical_groups(self):
        """Same values should give high p-value (not significant)."""
        _, p = _mann_whitney_u([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert p > 0.05

    def test_perfectly_separated(self):
        """Completely separated groups should give low p-value."""
        _, p = _mann_whitney_u(
            [0.1, 0.2, 0.15, 0.18, 0.12, 0.22, 0.13, 0.19],
            [0.8, 0.9, 0.85, 0.88, 0.82, 0.92, 0.83, 0.89],
        )
        assert p < 0.01

    def test_empty_group(self):
        u, p = _mann_whitney_u([], [1, 2, 3])
        assert p == 1.0

    def test_single_elements(self):
        u, p = _mann_whitney_u([1], [2])
        assert u == 1.0  # 1 < 2 → U = 1


class TestNormalCDF:
    def test_center(self):
        assert abs(_normal_cdf(0) - 0.5) < 0.001

    def test_extreme_positive(self):
        assert _normal_cdf(8.0) > 0.9999

    def test_extreme_negative(self):
        assert _normal_cdf(-8.0) < 0.0001

    def test_standard_values(self):
        # z=1.96 → ~0.975 (Abramowitz & Stegun approximation has ~0.5% error)
        assert abs(_normal_cdf(1.96) - 0.975) < 0.01


class TestBootstrapCI:
    def test_tight_ci_for_identical_values(self):
        import random
        rng = random.Random(42)
        lo, hi = _bootstrap_ci([0.5, 0.5, 0.5, 0.5], 1000, rng)
        assert lo == 0.5
        assert hi == 0.5

    def test_ci_contains_mean(self):
        import random
        rng = random.Random(42)
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        lo, hi = _bootstrap_ci(values, 5000, rng)
        mean = sum(values) / len(values)
        assert lo <= mean <= hi

    def test_single_value(self):
        import random
        rng = random.Random(42)
        lo, hi = _bootstrap_ci([0.7], 100, rng)
        assert lo == 0.7
        assert hi == 0.7


class TestOptimalThreshold:
    def test_perfect_separation(self):
        """Perfectly separated scores should give J=1.0."""
        scored = [(0.3, True), (0.4, True), (0.8, False), (0.9, False)]
        threshold, sens, spec, j = _optimal_threshold(scored)
        assert j >= 0.9

    def test_no_separation(self):
        """Identical scores should give J=0."""
        scored = [(0.5, True), (0.5, True), (0.5, False), (0.5, False)]
        threshold, sens, spec, j = _optimal_threshold(scored)
        assert j <= 0.5

    def test_empty(self):
        threshold, sens, spec, j = _optimal_threshold([])
        assert threshold == 0.5


class TestAnalyzeSeparation:
    def _make_score(self, instance_id, n_fp, composite):
        """Helper to create a VerifierScore with specific values."""
        candidates = [
            CandidatePatch("gold", "", "gold", verifier_pass=True, true_quality=PatchQuality.CORRECT),
        ]
        # Add exploit candidates based on n_fp
        for i in range(n_fp):
            candidates.append(CandidatePatch(
                f"exploit_{i}", "", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT,
            ))
        candidates.append(CandidatePatch("trivial", "", "trivial", verifier_pass=False, true_quality=PatchQuality.TRIVIAL))
        score = compute_verifier_score(instance_id, candidates)
        return score

    def test_significant_separation(self):
        hackable = [self._make_score(f"h{i}", 2, 0.0) for i in range(8)]
        non_hackable = [self._make_score(f"n{i}", 0, 0.0) for i in range(20)]
        sep = analyze_separation(hackable, non_hackable)
        assert sep.is_significant
        assert sep.delta > 0
        assert sep.effect_size_r != 0
