"""
Statistical analysis for verifier scoring validation.

Provides:
  - Mann-Whitney U test for hackable vs non-hackable score separation
  - Bootstrap confidence intervals for mean composite scores
  - Optimal threshold via Youden's J statistic
  - Weight sensitivity analysis
  - Per-exploit-category success rate analysis
  - Effect size (rank-biserial correlation)

No scipy/numpy dependency — all implemented from scratch.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from envaudit.scoring.verifier_scorer import VerifierScore, compute_verifier_score, CandidatePatch


@dataclass
class SeparationAnalysis:
    """Statistical analysis of hackable vs non-hackable score separation."""
    n_hackable: int
    n_non_hackable: int
    mean_hackable: float
    mean_non_hackable: float
    delta: float
    mann_whitney_u: float
    p_value: float
    effect_size_r: float           # Rank-biserial correlation (-1 to 1)
    ci_lower_hackable: float       # 95% CI for hackable mean
    ci_upper_hackable: float
    ci_lower_non_hackable: float
    ci_upper_non_hackable: float
    optimal_threshold: float       # Composite score that best separates groups
    threshold_sensitivity: float   # True positive rate at optimal threshold
    threshold_specificity: float   # True negative rate at optimal threshold
    threshold_youden_j: float      # Youden's J = sensitivity + specificity - 1
    is_significant: bool           # p < 0.05


@dataclass
class WeightSensitivity:
    """How composite score separation changes under different weight configurations."""
    weight_config_name: str
    weights: Dict[str, float]
    mean_hackable: float
    mean_non_hackable: float
    delta: float
    p_value: float


@dataclass
class CategoryAnalysis:
    """Per-exploit-category success rates from Docker verification."""
    category: str
    total_attempts: int
    docker_verified: int
    success_rate: float
    example_tasks: List[str]


def analyze_separation(
    hackable_scores: List[VerifierScore],
    non_hackable_scores: List[VerifierScore],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> SeparationAnalysis:
    """Full statistical analysis of hackable vs non-hackable separation."""
    h_vals = [s.composite_score for s in hackable_scores]
    n_vals = [s.composite_score for s in non_hackable_scores]

    mean_h = _mean(h_vals)
    mean_n = _mean(n_vals)

    # Mann-Whitney U test
    u_stat, p_val = _mann_whitney_u(h_vals, n_vals)

    # Rank-biserial correlation (effect size for Mann-Whitney)
    n1, n2 = len(h_vals), len(n_vals)
    r = 1.0 - (2.0 * u_stat) / (n1 * n2) if n1 * n2 > 0 else 0.0

    # Bootstrap 95% CIs
    rng = random.Random(seed)
    ci_h = _bootstrap_ci(h_vals, n_bootstrap, rng)
    ci_n = _bootstrap_ci(n_vals, n_bootstrap, rng)

    # Optimal threshold via Youden's J
    all_scores = [(s.composite_score, s.n_false_positives > 0) for s in hackable_scores + non_hackable_scores]
    threshold, sens, spec, j = _optimal_threshold(all_scores)

    return SeparationAnalysis(
        n_hackable=len(h_vals),
        n_non_hackable=len(n_vals),
        mean_hackable=round(mean_h, 4),
        mean_non_hackable=round(mean_n, 4),
        delta=round(mean_n - mean_h, 4),
        mann_whitney_u=round(u_stat, 2),
        p_value=round(p_val, 6),
        effect_size_r=round(r, 4),
        ci_lower_hackable=round(ci_h[0], 4),
        ci_upper_hackable=round(ci_h[1], 4),
        ci_lower_non_hackable=round(ci_n[0], 4),
        ci_upper_non_hackable=round(ci_n[1], 4),
        optimal_threshold=round(threshold, 4),
        threshold_sensitivity=round(sens, 4),
        threshold_specificity=round(spec, 4),
        threshold_youden_j=round(j, 4),
        is_significant=p_val < 0.05,
    )


def analyze_weight_sensitivity(
    hackable_scores: List[VerifierScore],
    non_hackable_scores: List[VerifierScore],
) -> List[WeightSensitivity]:
    """Test composite score separation under different weight configurations."""
    configs = {
        "default": {"discrimination": 0.40, "spearman": 0.20, "mae": 0.15, "completeness": 0.25},
        "discrimination_heavy": {"discrimination": 0.70, "spearman": 0.10, "mae": 0.10, "completeness": 0.10},
        "balanced": {"discrimination": 0.25, "spearman": 0.25, "mae": 0.25, "completeness": 0.25},
        "completeness_heavy": {"discrimination": 0.20, "spearman": 0.15, "mae": 0.15, "completeness": 0.50},
        "spearman_heavy": {"discrimination": 0.20, "spearman": 0.50, "mae": 0.15, "completeness": 0.15},
    }

    results = []
    for name, weights in configs.items():
        h_recomputed = [compute_verifier_score(s.instance_id, s.candidates, weights) for s in hackable_scores]
        n_recomputed = [compute_verifier_score(s.instance_id, s.candidates, weights) for s in non_hackable_scores]

        h_vals = [s.composite_score for s in h_recomputed]
        n_vals = [s.composite_score for s in n_recomputed]

        _, p_val = _mann_whitney_u(h_vals, n_vals)

        results.append(WeightSensitivity(
            weight_config_name=name,
            weights=weights,
            mean_hackable=round(_mean(h_vals), 4),
            mean_non_hackable=round(_mean(n_vals), 4),
            delta=round(_mean(n_vals) - _mean(h_vals), 4),
            p_value=round(p_val, 6),
        ))

    return results


def analyze_exploit_categories(
    scores: Dict[str, VerifierScore],
    phase1_data: List[Dict],
) -> List[CategoryAnalysis]:
    """Per-exploit-category success rates from Docker verification."""
    category_stats: Dict[str, Dict] = {}

    for task in phase1_data:
        instance_id = task["instance_id"]
        strategies = task.get("strategies", [])
        score = scores.get(instance_id)
        if not score:
            continue

        for i, strategy in enumerate(strategies):
            cat = strategy.get("category", "unknown")
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "verified": 0, "tasks": set()}

            # Find the corresponding candidate
            patch_id = f"{instance_id}_exploit_{i}"
            candidate = next((c for c in score.candidates if c.patch_id == patch_id), None)
            if candidate and candidate.verifier_pass is not None:
                category_stats[cat]["total"] += 1
                if candidate.verifier_pass:
                    category_stats[cat]["verified"] += 1
                    category_stats[cat]["tasks"].add(instance_id)

    results = []
    for cat, stats in sorted(category_stats.items(), key=lambda x: -x[1]["verified"]):
        results.append(CategoryAnalysis(
            category=cat,
            total_attempts=stats["total"],
            docker_verified=stats["verified"],
            success_rate=round(stats["verified"] / max(stats["total"], 1), 4),
            example_tasks=sorted(stats["tasks"])[:3],
        ))

    return results


# ─── Statistical Primitives (no scipy) ─────────────────────────────────


def _mann_whitney_u(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Mann-Whitney U test for two independent samples.

    Returns (U statistic, approximate two-sided p-value).
    Uses normal approximation for p-value (valid for n1, n2 >= 8).
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Compute U: count how many times x[i] < y[j]
    u = 0.0
    for xi in x:
        for yj in y:
            if xi < yj:
                u += 1.0
            elif xi == yj:
                u += 0.5

    # Normal approximation
    mu = n1 * n2 / 2.0
    # Tie correction
    all_vals = sorted(x + y)
    tie_correction = _tie_correction(all_vals)
    n = n1 + n2
    sigma_sq = (n1 * n2 / 12.0) * (n + 1 - tie_correction / (n * (n - 1)))
    sigma = math.sqrt(max(sigma_sq, 1e-10))

    z = (u - mu) / sigma
    p = 2.0 * _normal_cdf(-abs(z))

    return u, p


def _tie_correction(sorted_vals: List[float]) -> float:
    """Compute tie correction factor for Mann-Whitney."""
    correction = 0.0
    i = 0
    n = len(sorted_vals)
    while i < n:
        j = i
        while j < n - 1 and sorted_vals[j + 1] == sorted_vals[j]:
            j += 1
        t = j - i + 1
        if t > 1:
            correction += t * t * t - t
        i = j + 1
    return correction


def _normal_cdf(z: float) -> float:
    """Approximate standard normal CDF using Abramowitz & Stegun."""
    if z < -8.0:
        return 0.0
    if z > 8.0:
        return 1.0
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1.0 if z >= 0 else -1.0
    z_abs = abs(z)
    t = 1.0 / (1.0 + p * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs / 2.0)
    return 0.5 * (1.0 + sign * y)


def _bootstrap_ci(
    values: List[float],
    n_bootstrap: int,
    rng: random.Random,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap 95% confidence interval for the mean."""
    if len(values) < 2:
        m = _mean(values)
        return (m, m)

    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(_mean(sample))

    means.sort()
    lo = int(n_bootstrap * alpha / 2)
    hi = int(n_bootstrap * (1 - alpha / 2))
    return (means[lo], means[min(hi, len(means) - 1)])


def _optimal_threshold(
    scored_labels: List[Tuple[float, bool]],
) -> Tuple[float, float, float, float]:
    """Find optimal threshold via Youden's J statistic.

    Args:
        scored_labels: List of (composite_score, is_hackable) tuples.

    Returns:
        (threshold, sensitivity, specificity, youden_j)
        where hackable tasks should be BELOW the threshold.
    """
    if not scored_labels:
        return 0.5, 0.0, 0.0, 0.0

    positives = [s for s, label in scored_labels if label]
    negatives = [s for s, label in scored_labels if not label]
    n_pos = len(positives)
    n_neg = len(negatives)

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0, 0.0, 0.0

    # Try each unique score as threshold
    all_scores = sorted(set(s for s, _ in scored_labels))
    best_j = -1.0
    best_threshold = 0.5
    best_sens = 0.0
    best_spec = 0.0

    for threshold in all_scores:
        # Hackable tasks should score BELOW threshold
        tp = sum(1 for s in positives if s <= threshold)  # correctly identified as hackable
        tn = sum(1 for s in negatives if s > threshold)   # correctly identified as non-hackable
        sensitivity = tp / n_pos
        specificity = tn / n_neg
        j = sensitivity + specificity - 1.0

        if j > best_j:
            best_j = j
            best_threshold = threshold
            best_sens = sensitivity
            best_spec = specificity

    return best_threshold, best_sens, best_spec, best_j


def _mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)
