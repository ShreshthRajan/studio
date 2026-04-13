"""Tests for the Verifier Scoring module (NVIDIA framework)."""

import pytest
from envaudit.scoring.verifier_scorer import (
    CandidatePatch,
    PatchQuality,
    VerifierScore,
    compute_verifier_score,
    _spearman_correlation,
    _rank,
)
from envaudit.scoring.patch_assembler import _parse_docker_logs


# ─── Spearman Correlation Tests ────────────────────────────────────────


class TestSpearmanCorrelation:
    def test_perfect_positive(self):
        """Perfectly correlated rankings should give rho ≈ 1.0."""
        rho = _spearman_correlation([1, 2, 3, 4], [1, 2, 3, 4])
        assert abs(rho - 1.0) < 1e-10

    def test_perfect_negative(self):
        """Perfectly inverse rankings should give rho ≈ -1.0."""
        rho = _spearman_correlation([1, 2, 3, 4], [4, 3, 2, 1])
        assert abs(rho - (-1.0)) < 1e-10

    def test_no_correlation(self):
        """Uncorrelated should be near 0."""
        rho = _spearman_correlation([1, 2, 3, 4], [2, 4, 1, 3])
        assert -0.5 < rho < 0.5

    def test_identical_values(self):
        """All identical values should give 0 (no ranking possible)."""
        rho = _spearman_correlation([1, 1, 1], [1, 1, 1])
        assert rho == 0.0

    def test_single_element(self):
        rho = _spearman_correlation([1], [1])
        assert rho == 0.0

    def test_empty(self):
        rho = _spearman_correlation([], [])
        assert rho == 0.0

    def test_ties_handled(self):
        """Tied values should use average ranks."""
        rho = _spearman_correlation([1, 1, 2, 3], [1, 2, 3, 4])
        assert -1.0 <= rho <= 1.0


class TestRank:
    def test_basic_ranking(self):
        ranks = _rank([10, 30, 20])
        assert ranks == [1.0, 3.0, 2.0]

    def test_ties(self):
        ranks = _rank([10, 10, 30])
        assert ranks == [1.5, 1.5, 3.0]

    def test_all_tied(self):
        ranks = _rank([5, 5, 5])
        assert ranks == [2.0, 2.0, 2.0]


# ─── Verifier Score Computation Tests ──────────────────────────────────


class TestComputeVerifierScore:
    def test_perfect_verifier(self):
        """Verifier that perfectly discriminates correct from incorrect."""
        candidates = [
            CandidatePatch("gold", "", "gold", verifier_pass=True, true_quality=PatchQuality.CORRECT),
            CandidatePatch("exploit", "", "exploit", verifier_pass=False, true_quality=PatchQuality.EXPLOIT),
            CandidatePatch("trivial", "", "trivial", verifier_pass=False, true_quality=PatchQuality.TRIVIAL),
        ]
        score = compute_verifier_score("test-001", candidates)

        assert score.top1_accuracy == 1.0
        assert score.spearman_rho > 0.5
        assert score.n_false_positives == 0
        assert score.n_false_negatives == 0
        assert score.false_positive_rate == 0.0
        # With FPR=0 and FNR=0, composite should be high
        assert score.composite_score > 0.7

    def test_hackable_verifier(self):
        """Verifier that accepts an exploit — should score low."""
        candidates = [
            CandidatePatch("gold", "", "gold", verifier_pass=True, true_quality=PatchQuality.CORRECT),
            CandidatePatch("exploit", "", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT),
            CandidatePatch("trivial", "", "trivial", verifier_pass=False, true_quality=PatchQuality.TRIVIAL),
        ]
        score = compute_verifier_score("test-002", candidates)

        assert score.n_false_positives == 1
        assert score.false_positive_rate > 0.0
        # Top-1: both gold and exploit pass, so top1 may or may not be 1.0
        # But composite should be lower than perfect case
        assert score.composite_score < 1.0

    def test_overly_strict_verifier(self):
        """Verifier that rejects even the correct patch — false negative."""
        candidates = [
            CandidatePatch("gold", "", "gold", verifier_pass=False, true_quality=PatchQuality.CORRECT),
            CandidatePatch("exploit", "", "exploit", verifier_pass=False, true_quality=PatchQuality.EXPLOIT),
            CandidatePatch("trivial", "", "trivial", verifier_pass=False, true_quality=PatchQuality.TRIVIAL),
        ]
        score = compute_verifier_score("test-003", candidates)

        assert score.n_false_negatives == 1
        assert score.false_negative_rate == 1.0
        # Verifier gives same score (0.0) to everything — no discrimination
        # Spearman returns 0 when one variable is constant
        assert score.spearman_rho == 0.0
        # Completeness penalized (FNR=1.0)
        assert score.composite_score < 0.7

    def test_worst_verifier(self):
        """Verifier that accepts exploits but rejects correct solution."""
        candidates = [
            CandidatePatch("gold", "", "gold", verifier_pass=False, true_quality=PatchQuality.CORRECT),
            CandidatePatch("exploit", "", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT),
            CandidatePatch("trivial", "", "trivial", verifier_pass=True, true_quality=PatchQuality.TRIVIAL),
        ]
        score = compute_verifier_score("test-004", candidates)

        assert score.n_false_positives == 2
        assert score.n_false_negatives == 1
        assert score.spearman_rho < 0  # Inverted correlation
        assert score.composite_score < 0.5

    def test_empty_candidates(self):
        score = compute_verifier_score("test-empty", [])
        assert score.composite_score == 0.0
        assert score.n_candidates == 0

    def test_single_candidate(self):
        candidates = [
            CandidatePatch("gold", "", "gold", verifier_pass=True, true_quality=PatchQuality.CORRECT),
        ]
        score = compute_verifier_score("test-single", candidates)
        assert score.n_candidates == 1
        assert score.composite_score > 0.0

    def test_all_exploits_pass(self):
        """All exploits pass — verifier is completely broken."""
        candidates = [
            CandidatePatch("gold", "", "gold", verifier_pass=True, true_quality=PatchQuality.CORRECT),
            CandidatePatch("e1", "", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT),
            CandidatePatch("e2", "", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT),
            CandidatePatch("e3", "", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT),
            CandidatePatch("trivial", "", "trivial", verifier_pass=False, true_quality=PatchQuality.TRIVIAL),
        ]
        score = compute_verifier_score("test-broken", candidates)
        assert score.false_positive_rate == 0.75  # 3 out of 4 incorrect patches pass
        # With FPR=0.75, discrimination=0.25, composite should be low
        assert score.composite_score < 0.75

    def test_hackable_has_lower_score_than_robust(self):
        """Core validation: hackable verifier should score lower than robust one."""
        # Robust verifier
        robust_candidates = [
            CandidatePatch("gold", "", "gold", verifier_pass=True, true_quality=PatchQuality.CORRECT),
            CandidatePatch("e1", "", "exploit", verifier_pass=False, true_quality=PatchQuality.EXPLOIT),
            CandidatePatch("e2", "", "exploit", verifier_pass=False, true_quality=PatchQuality.EXPLOIT),
            CandidatePatch("trivial", "", "trivial", verifier_pass=False, true_quality=PatchQuality.TRIVIAL),
        ]
        robust_score = compute_verifier_score("robust", robust_candidates)

        # Hackable verifier
        hackable_candidates = [
            CandidatePatch("gold", "", "gold", verifier_pass=True, true_quality=PatchQuality.CORRECT),
            CandidatePatch("e1", "", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT),
            CandidatePatch("e2", "", "exploit", verifier_pass=True, true_quality=PatchQuality.EXPLOIT),
            CandidatePatch("trivial", "", "trivial", verifier_pass=False, true_quality=PatchQuality.TRIVIAL),
        ]
        hackable_score = compute_verifier_score("hackable", hackable_candidates)

        assert robust_score.composite_score > hackable_score.composite_score


# ─── Patch Assembler Tests ─────────────────────────────────────────────


class TestParsDockerLogs:
    def test_missing_directory(self, tmp_path):
        results = _parse_docker_logs(str(tmp_path / "nonexistent"))
        assert results == {}

    def test_parses_resolved(self, tmp_path):
        log = tmp_path / "django__django-11276_exploit_0.log"
        log.write_text("Some output\nInstances resolved: 1\nDone")
        results = _parse_docker_logs(str(tmp_path))
        assert results["django__django-11276_exploit_0"] is True

    def test_parses_unresolved(self, tmp_path):
        log = tmp_path / "django__django-11276_exploit_1.log"
        log.write_text("Some output\nInstances resolved: 0\nDone")
        results = _parse_docker_logs(str(tmp_path))
        assert results["django__django-11276_exploit_1"] is False

    def test_multiple_logs(self, tmp_path):
        (tmp_path / "task_exploit_0.log").write_text("Instances resolved: 1")
        (tmp_path / "task_exploit_1.log").write_text("Instances resolved: 0")
        results = _parse_docker_logs(str(tmp_path))
        assert len(results) == 2
        assert results["task_exploit_0"] is True
        assert results["task_exploit_1"] is False


# ─── Integration Test: Full Pipeline ───────────────────────────────────


class TestIntegration:
    def test_scoring_pipeline_with_mock_data(self, tmp_path):
        """End-to-end test with synthetic phase1 results and Docker logs."""
        # Create mock phase1 results
        phase1 = [
            {
                "instance_id": "test__test-hackable",
                "verdict": "FAIL",
                "score": 0.1,
                "confidence": 0.9,
                "n_pattern_flags": 2,
                "n_strategies": 2,
                "max_exploit_confidence": 9,
                "strategies": [
                    {"name": "exploit1", "category": "test_suite_exploitation", "confidence": 9, "description": "desc1"},
                    {"name": "exploit2", "category": "solution_degradation", "confidence": 8, "description": "desc2"},
                ],
                "exploit_patches": ["patch1_content", "patch2_content"],
                "cost_usd": 0.02,
            },
            {
                "instance_id": "test__test-robust",
                "verdict": "FAIL",
                "score": 0.1,
                "confidence": 0.9,
                "n_pattern_flags": 1,
                "n_strategies": 2,
                "max_exploit_confidence": 8,
                "strategies": [
                    {"name": "exploit1", "category": "test_suite_exploitation", "confidence": 8, "description": "desc1"},
                    {"name": "exploit2", "category": "context_exploitation", "confidence": 7, "description": "desc2"},
                ],
                "exploit_patches": ["patch3_content", "patch4_content"],
                "cost_usd": 0.02,
            },
        ]
        phase1_path = tmp_path / "phase1_results.json"
        phase1_path.write_text(json.dumps(phase1))

        # Create mock Docker logs
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "test__test-hackable_exploit_0.log").write_text("Instances resolved: 1")
        (log_dir / "test__test-hackable_exploit_1.log").write_text("Instances resolved: 1")
        (log_dir / "test__test-robust_exploit_0.log").write_text("Instances resolved: 0")
        (log_dir / "test__test-robust_exploit_1.log").write_text("Instances resolved: 0")

        # Gold patches
        gold_patches = {
            "test__test-hackable": "gold_patch_hackable",
            "test__test-robust": "gold_patch_robust",
        }

        # Assemble candidates
        from envaudit.scoring.patch_assembler import assemble_candidates_from_phase1
        task_candidates = assemble_candidates_from_phase1(
            str(phase1_path), str(log_dir), gold_patches
        )

        assert len(task_candidates) == 2
        assert len(task_candidates["test__test-hackable"]) == 4  # gold + 2 exploits + trivial
        assert len(task_candidates["test__test-robust"]) == 4

        # Compute scores
        hackable_score = compute_verifier_score(
            "test__test-hackable", task_candidates["test__test-hackable"]
        )
        robust_score = compute_verifier_score(
            "test__test-robust", task_candidates["test__test-robust"]
        )

        # Hackable task (2/2 exploits pass) should score LOWER than
        # robust task (0/2 exploits pass)
        assert hackable_score.composite_score < robust_score.composite_score
        assert hackable_score.n_false_positives == 2
        assert robust_score.n_false_positives == 0


# Need json for integration test
import json
