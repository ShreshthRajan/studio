"""
Hybrid Verification — combines execution-based (test suite) and
judgment-based (semi-formal reasoning) verification signals.

Based on: R2E-Gym (COLM 2025) — hybrid of execution + LLM judge
gives +8pp over either alone.

Catches BOTH types of verifier failure:
  - False positives: tests pass but judge says incorrect (hackable)
  - False negatives: tests fail but judge says correct (overly strict)

This is the ONLY component that catches overly-strict tests — what
OpenAI found in 59.4% of SWE-bench Verified tasks.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from envaudit.scoring.verifier_scorer import CandidatePatch, PatchQuality
from envaudit.scoring.semiformal_judge import CorrectnessVerdict, JudgmentResult

logger = logging.getLogger(__name__)


@dataclass
class ConfusionEntry:
    """One cell in the confusion matrix for a single candidate patch."""
    patch_id: str
    verifier_pass: bool          # Test suite says: pass (True) or fail (False)
    judge_correct: bool          # Semi-formal judge says: correct (True) or not (False)
    category: str                # "TP", "FP", "FN", "TN"


@dataclass
class HybridResult:
    """Per-task hybrid verification result with confusion matrix."""
    instance_id: str
    entries: List[ConfusionEntry]
    tp: int                      # Tests pass + judge correct
    fp: int                      # Tests pass + judge incorrect (HACKABLE)
    fn: int                      # Tests fail + judge correct (OVERLY STRICT)
    tn: int                      # Tests fail + judge incorrect
    precision: float             # TP / (TP + FP) — how many passing patches are actually correct
    recall: float                # TP / (TP + FN) — how many correct patches pass tests
    f1: float                    # Harmonic mean of precision and recall
    false_positive_rate: float   # FP / (FP + TN) — hackability signal
    false_negative_rate: float   # FN / (FN + TP) — over-strictness signal
    n_judged: int                # Total candidates that were judged
    n_skipped: int               # Candidates skipped (no verifier_pass or uncertain judge)
    total_judge_cost: float
    metadata: Dict = field(default_factory=dict)


def compute_hybrid_result(
    instance_id: str,
    candidates: List[CandidatePatch],
    judgments: Dict[str, JudgmentResult],
) -> HybridResult:
    """Compute hybrid verification confusion matrix for a task.

    Args:
        instance_id: Task identifier.
        candidates: Candidate patches with verifier_pass set.
        judgments: Dict mapping patch_id → JudgmentResult from semi-formal judge.

    Returns:
        HybridResult with confusion matrix and derived metrics.
    """
    entries = []
    tp = fp = fn = tn = 0
    n_skipped = 0
    total_cost = sum(j.cost_usd for j in judgments.values())

    for c in candidates:
        judgment = judgments.get(c.patch_id)
        if judgment is None or c.verifier_pass is None:
            n_skipped += 1
            continue

        if judgment.verdict == CorrectnessVerdict.UNCERTAIN:
            n_skipped += 1
            continue

        # Binary: judge says correct (CORRECT or PARTIAL) vs incorrect
        judge_correct = judgment.verdict in (
            CorrectnessVerdict.CORRECT,
            CorrectnessVerdict.PARTIAL,
        )
        verifier_pass = c.verifier_pass

        if verifier_pass and judge_correct:
            category = "TP"
            tp += 1
        elif verifier_pass and not judge_correct:
            category = "FP"
            fp += 1
        elif not verifier_pass and judge_correct:
            category = "FN"
            fn += 1
        else:
            category = "TN"
            tn += 1

        entries.append(ConfusionEntry(
            patch_id=c.patch_id,
            verifier_pass=verifier_pass,
            judge_correct=judge_correct,
            category=category,
        ))

    n_judged = len(entries)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    return HybridResult(
        instance_id=instance_id,
        entries=entries,
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        false_negative_rate=round(fnr, 4),
        n_judged=n_judged,
        n_skipped=n_skipped,
        total_judge_cost=round(total_cost, 6),
    )


def upgrade_candidate_quality(
    candidate: CandidatePatch,
    judgment: JudgmentResult,
) -> PatchQuality:
    """Upgrade a candidate's true_quality based on semi-formal judgment.

    This replaces the heuristic source-based quality assignment with
    a principled judgment-based one.

    Priority logic:
    - If Docker verified AND judge agrees → high confidence in the label
    - If Docker and judge disagree → trust Docker for pass/fail but flag the disagreement
    - If no Docker data → use judge verdict directly
    """
    docker_passed = candidate.verifier_pass

    if judgment.verdict == CorrectnessVerdict.UNCERTAIN:
        # Judge can't decide — keep existing quality label
        return candidate.true_quality

    judge_says_correct = judgment.verdict in (
        CorrectnessVerdict.CORRECT,
        CorrectnessVerdict.PARTIAL,
    )

    if candidate.source == "gold":
        # Gold patch is always CORRECT regardless of judge
        return PatchQuality.CORRECT

    if candidate.source == "trivial":
        # Empty patch is always TRIVIAL regardless of judge
        return PatchQuality.TRIVIAL

    # For exploit and other sources:
    if judge_says_correct:
        if judgment.verdict == CorrectnessVerdict.CORRECT:
            return PatchQuality.CORRECT
        else:
            return PatchQuality.PARTIAL
    else:
        # Judge says incorrect
        if docker_passed:
            # Tests pass but judge says wrong → EXPLOIT (confirmed hackable)
            return PatchQuality.EXPLOIT
        else:
            # Tests fail AND judge says wrong → TRIVIAL
            return PatchQuality.TRIVIAL
