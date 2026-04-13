"""
Verifier Scorer — quantifies how well a task's test suite discriminates
correct from incorrect solutions.

Based on: "Scoring Verifiers: Evaluating Synthetic Verification for Code
and Reasoning" (NVIDIA, Feb 2025, arXiv:2502.13820).

The NVIDIA framework proposes 4 metrics:
  - Top-1 Accuracy: Does the verifier rank the best patch highest?
  - Bottom-1 Accuracy: Does the verifier reject the worst patch?
  - Spearman rho: Rank correlation between verifier score and true quality.
  - MAE: Mean absolute error between verifier score and true quality.

We adapt this for SWE-bench: the "verifier" is the test suite (pass/fail),
and "true quality" is assessed by comparing patches to the gold patch +
independent LLM correctness judgment.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class PatchQuality(IntEnum):
    """Ground-truth quality tier for a candidate patch.

    Higher value = better patch. Used as the reference ranking
    for computing Spearman correlation with verifier ranking.
    """
    TRIVIAL = 0       # Empty or no-op patch
    EXPLOIT = 1       # Passes tests but doesn't solve the problem
    PARTIAL = 2       # Addresses part of the problem
    CORRECT = 3       # Gold patch or semantic equivalent


@dataclass
class CandidatePatch:
    """A single candidate patch with both verifier outcome and true quality."""
    patch_id: str
    patch_text: str
    source: str                       # "gold", "exploit", "mutation", "trivial"
    verifier_pass: Optional[bool]     # Did the test suite say this passes?
    true_quality: PatchQuality        # Independent assessment of actual correctness
    verifier_score: float = 0.0       # Normalized verifier score (0.0 = fail, 1.0 = pass)
    true_score: float = 0.0           # Normalized true quality (0.0 = trivial, 1.0 = correct)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierScore:
    """Per-task verifier quality metrics (NVIDIA framework)."""
    instance_id: str
    top1_accuracy: float          # 1.0 if verifier ranks best patch highest
    bottom1_accuracy: float       # 1.0 if verifier rejects worst patch
    spearman_rho: float           # Rank correlation (-1 to 1)
    mae: float                    # Mean absolute error (0 to 1)
    composite_score: float        # Weighted combination
    n_candidates: int             # How many patches were evaluated
    n_false_positives: int        # Patches that pass tests but are incorrect
    n_false_negatives: int        # Patches that fail tests but are correct
    false_positive_rate: float    # n_false_positives / n_incorrect_patches
    false_negative_rate: float    # n_false_negatives / n_correct_patches
    candidates: List[CandidatePatch] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_verifier_score(
    instance_id: str,
    candidates: List[CandidatePatch],
    weights: Optional[Dict[str, float]] = None,
) -> VerifierScore:
    """Compute NVIDIA verifier metrics for a set of candidate patches.

    Adapts the NVIDIA framework for binary verifiers (test pass/fail).
    The composite score combines NVIDIA metrics with the false positive
    rate, which is the most direct measure of hackability for binary
    verifiers.

    Args:
        instance_id: SWE-bench task identifier.
        candidates: Patches with both verifier outcomes and true quality labels.
        weights: Optional weights for composite score components.

    Returns:
        VerifierScore with all 4 NVIDIA metrics + false positive/negative rates.
    """
    if not candidates:
        return _empty_score(instance_id)

    weights = weights or {
        "discrimination": 0.40,  # 1 - FPR: can the verifier reject bad patches?
        "spearman": 0.20,        # Rank correlation
        "mae": 0.15,             # Score accuracy
        "completeness": 0.25,    # 1 - FNR: does the verifier accept good patches?
    }

    # Normalize scores for each candidate
    for c in candidates:
        c.verifier_score = 1.0 if c.verifier_pass else 0.0
        c.true_score = c.true_quality.value / PatchQuality.CORRECT.value

    # --- Top-1 Accuracy ---
    # Does the verifier assign highest score to the best true-quality patch?
    best_true = max(candidates, key=lambda c: (c.true_quality, c.patch_id))
    best_verifier = max(candidates, key=lambda c: (c.verifier_score, c.true_quality))
    top1 = 1.0 if best_verifier.true_quality == best_true.true_quality else 0.0

    # --- Bottom-1 Accuracy ---
    # Does the verifier assign lowest score to the worst true-quality patch?
    worst_true = min(candidates, key=lambda c: (c.true_quality, c.patch_id))
    worst_verifier = min(candidates, key=lambda c: (c.verifier_score, -c.true_quality.value))
    bottom1 = 1.0 if worst_verifier.true_quality == worst_true.true_quality else 0.0

    # --- Spearman Rank Correlation ---
    spearman = _spearman_correlation(
        [c.verifier_score for c in candidates],
        [c.true_score for c in candidates],
    )

    # --- MAE ---
    mae = sum(abs(c.verifier_score - c.true_score) for c in candidates) / len(candidates)

    # --- False Positive / False Negative rates ---
    # FP: incorrect patches that the verifier accepts (hackability signal)
    # FN: correct patches that the verifier rejects (over-strictness signal)
    incorrect_patches = [c for c in candidates if c.true_quality <= PatchQuality.EXPLOIT]
    correct_patches = [c for c in candidates if c.true_quality >= PatchQuality.CORRECT]
    n_fp = sum(1 for c in incorrect_patches if c.verifier_pass)
    n_fn = sum(1 for c in correct_patches if not c.verifier_pass)
    fpr = n_fp / max(len(incorrect_patches), 1)
    fnr = n_fn / max(len(correct_patches), 1)

    # --- Composite Score ---
    # For binary verifiers, FPR is the most direct hackability measure.
    # A verifier that accepts all exploits (FPR=1.0) gets composite=0.
    # A verifier that rejects all exploits (FPR=0.0) gets high composite.
    discrimination = 1.0 - fpr       # Higher = better at rejecting bad patches
    completeness = 1.0 - fnr         # Higher = better at accepting good patches
    mae_inverted = 1.0 - mae
    spearman_norm = (spearman + 1.0) / 2.0

    composite = (
        weights["discrimination"] * discrimination
        + weights["spearman"] * spearman_norm
        + weights["mae"] * mae_inverted
        + weights["completeness"] * completeness
    )

    return VerifierScore(
        instance_id=instance_id,
        top1_accuracy=top1,
        bottom1_accuracy=bottom1,
        spearman_rho=round(spearman, 4),
        mae=round(mae, 4),
        composite_score=round(composite, 4),
        n_candidates=len(candidates),
        n_false_positives=n_fp,
        n_false_negatives=n_fn,
        false_positive_rate=round(fpr, 4),
        false_negative_rate=round(fnr, 4),
        candidates=candidates,
        metadata={
            "weights": weights,
            "n_correct": len(correct_patches),
            "n_incorrect": len(incorrect_patches),
        },
    )


def _spearman_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation without scipy dependency.

    Uses the standard formula: rho = 1 - 6*sum(d_i^2) / (n*(n^2-1))
    with average-rank tie handling.
    """
    n = len(x)
    if n < 2:
        return 0.0

    rx = _rank(x)
    ry = _rank(y)

    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    denom = n * (n * n - 1)
    if denom == 0:
        return 0.0

    rho = 1.0 - 6.0 * d_sq / denom

    # When all values in one list are tied, ranks are all identical,
    # so d_sq=0 and rho=1.0 — but there's actually no information.
    # Detect this and return 0.0 (no correlation).
    if len(set(x)) == 1 or len(set(y)) == 1:
        return 0.0

    return rho


def _rank(values: List[float]) -> List[float]:
    """Compute average ranks with tie handling."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1

    return ranks


def _empty_score(instance_id: str) -> VerifierScore:
    """Return a default score when no candidates are available."""
    return VerifierScore(
        instance_id=instance_id,
        top1_accuracy=0.0,
        bottom1_accuracy=0.0,
        spearman_rho=0.0,
        mae=1.0,
        composite_score=0.0,
        n_candidates=0,
        n_false_positives=0,
        n_false_negatives=0,
        false_positive_rate=0.0,
        false_negative_rate=0.0,
    )
