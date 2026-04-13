"""
Composite Environment Quality Score (EQS) — combines all scoring dimensions
into a single per-task quality metric with actionable verdicts.

Dimensions:
  1. Verifier Discrimination (Step 1): Can the test suite tell correct from incorrect?
  2. Exploit Resistance (Step 2): What fraction of adversarial exploits does the test suite block?
  3. Hybrid F1 (Step 3): How well does the verifier agree with an independent judge?
  4. Difficulty (Step 4): Is the task at the right difficulty for the target model?

Output per task:
  - EQS score (0.0 to 1.0)
  - Verdict: KEEP / FIX / DROP
  - Specific weaknesses identified
  - Recommended actions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

from envaudit.scoring.verifier_scorer import VerifierScore
from envaudit.scoring.hybrid import HybridResult
from envaudit.scoring.difficulty import DifficultyProfile

logger = logging.getLogger(__name__)


# Default weights — tunable, validated against training outcomes in Step 7
DEFAULT_WEIGHTS = {
    "verifier": 0.35,       # How well tests discriminate correct from incorrect
    "exploit": 0.30,        # Resistance to adversarial exploits
    "hybrid": 0.20,         # Agreement between tests and independent judge
    "difficulty": 0.15,     # Appropriate difficulty for training
}


@dataclass
class Weakness:
    """A specific weakness identified in a task's environment."""
    dimension: str           # "verifier", "exploit", "hybrid", "difficulty"
    severity: str            # "critical", "high", "medium", "low"
    description: str
    fixable: bool           # Can this be addressed by test augmentation?
    action: str             # Specific recommended action


@dataclass
class CompositeResult:
    """Per-task composite environment quality score."""
    instance_id: str
    eqs: float                           # Environment Quality Score (0.0 to 1.0)
    verdict: str                         # "KEEP", "FIX", "DROP"
    dimension_scores: Dict[str, float]   # Per-dimension scores (0.0 to 1.0)
    weaknesses: List[Weakness]
    weights: Dict[str, float]

    # Raw inputs (optional, for downstream use)
    verifier_score: Optional[VerifierScore] = None
    hybrid_result: Optional[HybridResult] = None
    difficulty_profile: Optional[DifficultyProfile] = None


def compute_eqs(
    instance_id: str,
    verifier: Optional[VerifierScore] = None,
    hybrid: Optional[HybridResult] = None,
    difficulty: Optional[DifficultyProfile] = None,
    exploit_success_rate: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> CompositeResult:
    """Compute the composite Environment Quality Score for a task.

    Any dimension can be None if data isn't available — the weights
    are renormalized across available dimensions.

    Args:
        instance_id: Task identifier.
        verifier: Step 1 verifier scoring result.
        hybrid: Step 3 hybrid verification result.
        difficulty: Step 4 difficulty profile.
        exploit_success_rate: Fraction of exploits that passed Docker (0.0 to 1.0).
        weights: Custom weights per dimension.
    """
    weights = weights or DEFAULT_WEIGHTS.copy()
    dim_scores: Dict[str, float] = {}
    weaknesses: List[Weakness] = []

    # --- Dimension 1: Verifier Discrimination ---
    if verifier is not None:
        dim_scores["verifier"] = verifier.composite_score
        weaknesses.extend(_assess_verifier_weaknesses(verifier))

    # --- Dimension 2: Exploit Resistance ---
    if exploit_success_rate is not None:
        dim_scores["exploit"] = 1.0 - exploit_success_rate
        weaknesses.extend(_assess_exploit_weaknesses(exploit_success_rate))

    # --- Dimension 3: Hybrid F1 ---
    if hybrid is not None:
        dim_scores["hybrid"] = hybrid.f1
        weaknesses.extend(_assess_hybrid_weaknesses(hybrid))

    # --- Dimension 4: Difficulty ---
    if difficulty is not None:
        # Normalize gradient signal to [0, 1] — max p(1-p) = 0.25 at p=0.5
        dim_scores["difficulty"] = min(difficulty.gradient_signal / 0.25, 1.0)
        weaknesses.extend(_assess_difficulty_weaknesses(difficulty))

    # --- Compute EQS ---
    if not dim_scores:
        return CompositeResult(
            instance_id=instance_id,
            eqs=0.0, verdict="DROP",
            dimension_scores={}, weaknesses=[],
            weights=weights,
        )

    # Renormalize weights to available dimensions
    available_weight_sum = sum(weights[d] for d in dim_scores)
    eqs = sum(
        dim_scores[d] * weights[d] / available_weight_sum
        for d in dim_scores
    )
    eqs = round(eqs, 4)

    # --- Verdict ---
    verdict = _compute_verdict(eqs, weaknesses)

    return CompositeResult(
        instance_id=instance_id,
        eqs=eqs,
        verdict=verdict,
        dimension_scores={d: round(s, 4) for d, s in dim_scores.items()},
        weaknesses=weaknesses,
        weights=weights,
        verifier_score=verifier,
        hybrid_result=hybrid,
        difficulty_profile=difficulty,
    )


def _compute_verdict(eqs: float, weaknesses: List[Weakness]) -> str:
    """Determine KEEP/FIX/DROP verdict with veto logic.

    Critical weaknesses force DROP regardless of EQS.
    Fixable weaknesses with moderate EQS → FIX.
    """
    # Veto: any critical unfixable weakness → DROP
    has_critical_unfixable = any(
        w.severity == "critical" and not w.fixable
        for w in weaknesses
    )
    if has_critical_unfixable:
        return "DROP"

    # Score-based thresholds
    if eqs >= 0.70:
        return "KEEP"
    elif eqs >= 0.40:
        has_fixable = any(w.fixable for w in weaknesses)
        return "FIX" if has_fixable else "DROP"
    else:
        return "DROP"


def _assess_verifier_weaknesses(v: VerifierScore) -> List[Weakness]:
    """Identify weaknesses from verifier scoring."""
    weaknesses = []

    if v.false_positive_rate > 0.5:
        weaknesses.append(Weakness(
            dimension="verifier",
            severity="critical",
            description=f"Verifier accepts {v.false_positive_rate:.0%} of incorrect patches",
            fixable=True,
            action="Add more discriminative test assertions targeting the specific exploit patterns",
        ))
    elif v.false_positive_rate > 0.0:
        weaknesses.append(Weakness(
            dimension="verifier",
            severity="high",
            description=f"Verifier accepts {v.false_positive_rate:.0%} of incorrect patches",
            fixable=True,
            action="Add test cases that distinguish the exploit from the gold patch",
        ))

    if v.false_negative_rate > 0.5:
        weaknesses.append(Weakness(
            dimension="verifier",
            severity="high",
            description=f"Verifier rejects {v.false_negative_rate:.0%} of correct patches",
            fixable=True,
            action="Relax overly-specific assertions that enforce implementation details",
        ))

    return weaknesses


def _assess_exploit_weaknesses(exploit_rate: float) -> List[Weakness]:
    """Identify weaknesses from exploit success rate."""
    weaknesses = []

    if exploit_rate > 0.5:
        weaknesses.append(Weakness(
            dimension="exploit",
            severity="critical",
            description=f"{exploit_rate:.0%} of adversarial exploits pass the test suite",
            fixable=True,
            action="Test suite needs fundamental strengthening — exploits consistently bypass it",
        ))
    elif exploit_rate > 0.0:
        weaknesses.append(Weakness(
            dimension="exploit",
            severity="high",
            description=f"{exploit_rate:.0%} of adversarial exploits pass the test suite",
            fixable=True,
            action="Add blocking tests for the specific exploit strategies that succeeded",
        ))

    return weaknesses


def _assess_hybrid_weaknesses(h: HybridResult) -> List[Weakness]:
    """Identify weaknesses from hybrid verification."""
    weaknesses = []

    if h.false_positive_rate > 0.3:
        weaknesses.append(Weakness(
            dimension="hybrid",
            severity="high",
            description=f"Judge disagrees with tests on {h.fp} patches (tests too permissive)",
            fixable=True,
            action="Tests accept patches that an independent judge deems incorrect",
        ))

    if h.false_negative_rate > 0.3:
        weaknesses.append(Weakness(
            dimension="hybrid",
            severity="medium",
            description=f"Tests reject {h.fn} patches that an independent judge deems correct",
            fixable=True,
            action="Tests may enforce specific implementation details — consider relaxing",
        ))

    return weaknesses


def _assess_difficulty_weaknesses(d: DifficultyProfile) -> List[Weakness]:
    """Identify weaknesses from difficulty profiling."""
    weaknesses = []

    if d.flag == "too_hard":
        weaknesses.append(Weakness(
            dimension="difficulty",
            severity="high",
            description=f"Task too hard (solve rate {d.solve_rate:.1%}) — near-zero gradient signal",
            fixable=False,
            action="Remove from training set or simplify the task",
        ))
    elif d.flag == "too_easy":
        weaknesses.append(Weakness(
            dimension="difficulty",
            severity="medium",
            description=f"Task too easy (solve rate {d.solve_rate:.1%}) — near-zero gradient signal",
            fixable=False,
            action="Remove from training set or increase task complexity",
        ))

    return weaknesses
