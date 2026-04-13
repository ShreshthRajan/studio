"""
Step 1: Compute Verifier Scores for all 49 tasks with Docker verification data.

Uses the NVIDIA "Scoring Verifiers" framework to quantify how well each task's
test suite discriminates correct from incorrect solutions.

Inputs:
  - experiments/phase1_results.json (50 tasks with exploit patches)
  - phase1_verification/verification_output/*.log (Docker results for 127 exploits)
  - SWE-bench Verified gold patches (from HuggingFace)

Outputs:
  - experiments/verifier_scores.json (per-task NVIDIA metrics)
  - Console summary with validation against Docker ground truth
"""

import json
import sys
import os
import logging
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envaudit.scoring.verifier_scorer import compute_verifier_score, VerifierScore
from envaudit.scoring.patch_assembler import (
    assemble_candidates_from_phase1,
    load_gold_patches,
)
from envaudit.scoring.statistics import (
    analyze_separation,
    analyze_weight_sensitivity,
    analyze_exploit_categories,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PHASE1_RESULTS = PROJECT_ROOT / "experiments" / "phase1_results.json"
VERIFICATION_LOGS = PROJECT_ROOT / "phase1_verification" / "verification_output"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "verifier_scores.json"


def main():
    parser = argparse.ArgumentParser(description="Compute verifier scores for SWE-bench tasks")
    parser.add_argument("--skip-gold", action="store_true",
                        help="Skip loading gold patches from HuggingFace (faster, less accurate)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="Output path for verifier scores JSON")
    args = parser.parse_args()

    # --- Load data ---
    logger.info("Loading phase1 results from %s", PHASE1_RESULTS)
    if not PHASE1_RESULTS.exists():
        logger.error("phase1_results.json not found at %s", PHASE1_RESULTS)
        sys.exit(1)

    gold_patches = None
    if not args.skip_gold:
        logger.info("Loading gold patches from SWE-bench Verified...")
        try:
            gold_patches = load_gold_patches()
            logger.info("Loaded %d gold patches", len(gold_patches))
        except Exception as e:
            logger.warning("Failed to load gold patches: %s. Continuing without them.", e)

    # --- Assemble candidates ---
    logger.info("Assembling candidate patches from phase1 + Docker logs...")
    task_candidates = assemble_candidates_from_phase1(
        phase1_path=str(PHASE1_RESULTS),
        verification_log_dir=str(VERIFICATION_LOGS),
        gold_patches=gold_patches,
    )
    logger.info("Assembled candidates for %d tasks", len(task_candidates))

    # --- Compute verifier scores ---
    scores: dict[str, VerifierScore] = {}
    for instance_id, candidates in sorted(task_candidates.items()):
        # Only score tasks that have Docker verification data
        docker_verified = [c for c in candidates if c.verifier_pass is not None]
        if not docker_verified and not gold_patches:
            logger.debug("Skipping %s — no Docker data and no gold patches", instance_id)
            continue

        score = compute_verifier_score(instance_id, candidates)
        scores[instance_id] = score

    logger.info("Computed verifier scores for %d tasks", len(scores))

    # --- Validate against Docker ground truth ---
    hackable = [s for s in scores.values() if s.n_false_positives > 0]
    non_hackable = [s for s in scores.values() if s.n_false_positives == 0]

    _print_validation(scores, hackable, non_hackable)

    # --- Statistical analysis ---
    sep = None
    if hackable and non_hackable:
        sep = analyze_separation(hackable, non_hackable)
        _print_statistical_analysis(sep)

        sensitivity = analyze_weight_sensitivity(hackable, non_hackable)
        _print_weight_sensitivity(sensitivity)

    # --- Exploit category analysis ---
    phase1_path = str(PHASE1_RESULTS)
    with open(phase1_path) as f:
        phase1_data = json.load(f)
    categories = analyze_exploit_categories(scores, phase1_data)
    _print_category_analysis(categories)

    # --- Save results ---
    output = {
        "total_tasks_scored": len(scores),
        "n_hackable": len(hackable),
        "n_non_hackable": len(non_hackable),
        "mean_composite": round(_mean([s.composite_score for s in scores.values()]), 4),
        "mean_spearman": round(_mean([s.spearman_rho for s in scores.values()]), 4),
        "mean_fpr": round(_mean([s.false_positive_rate for s in scores.values()]), 4),
        "mean_fnr": round(_mean([s.false_negative_rate for s in scores.values()]), 4),
        "statistical_analysis": {
            "mann_whitney_u": sep.mann_whitney_u,
            "p_value": sep.p_value,
            "effect_size_r": sep.effect_size_r,
            "is_significant": sep.is_significant,
            "optimal_threshold": sep.optimal_threshold,
            "threshold_youden_j": sep.threshold_youden_j,
            "ci_hackable": [sep.ci_lower_hackable, sep.ci_upper_hackable],
            "ci_non_hackable": [sep.ci_lower_non_hackable, sep.ci_upper_non_hackable],
        } if hackable and non_hackable else None,
        "exploit_categories": [
            {
                "category": c.category,
                "total_attempts": c.total_attempts,
                "docker_verified": c.docker_verified,
                "success_rate": c.success_rate,
            }
            for c in categories
        ],
        "per_task": {
            iid: {
                "top1_accuracy": s.top1_accuracy,
                "bottom1_accuracy": s.bottom1_accuracy,
                "spearman_rho": s.spearman_rho,
                "mae": s.mae,
                "composite_score": s.composite_score,
                "n_candidates": s.n_candidates,
                "n_false_positives": s.n_false_positives,
                "n_false_negatives": s.n_false_negatives,
                "false_positive_rate": s.false_positive_rate,
                "false_negative_rate": s.false_negative_rate,
                "candidates": [
                    {
                        "patch_id": c.patch_id,
                        "source": c.source,
                        "verifier_pass": c.verifier_pass,
                        "true_quality": c.true_quality.name,
                        "verifier_score": c.verifier_score,
                        "true_score": c.true_score,
                    }
                    for c in s.candidates
                ],
            }
            for iid, s in sorted(scores.items())
        },
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved verifier scores to %s", output_path)


def _print_validation(scores, hackable, non_hackable):
    """Print hackable vs non-hackable comparison."""
    print("\n" + "=" * 70)
    print("VERIFIER SCORING VALIDATION")
    print("=" * 70)
    print(f"\nTasks scored: {len(scores)}")
    print(f"Docker-verified hackable: {len(hackable)}")
    print(f"Non-hackable: {len(non_hackable)}")

    if hackable and non_hackable:
        hack_composite = _mean([s.composite_score for s in hackable])
        safe_composite = _mean([s.composite_score for s in non_hackable])
        hack_fpr = _mean([s.false_positive_rate for s in hackable])
        safe_fpr = _mean([s.false_positive_rate for s in non_hackable])

        print(f"\n{'Metric':<25} {'Hackable':>12} {'Non-hackable':>12} {'Delta':>10}")
        print("-" * 60)
        print(f"{'Composite Score':<25} {hack_composite:>12.4f} {safe_composite:>12.4f} {safe_composite - hack_composite:>+10.4f}")
        print(f"{'False Positive Rate':<25} {hack_fpr:>12.4f} {safe_fpr:>12.4f} {safe_fpr - hack_fpr:>+10.4f}")

        if hack_composite < safe_composite:
            print(f"\n✓ VALIDATED: Hackable tasks have lower verifier scores ({hack_composite:.4f} < {safe_composite:.4f})")
        else:
            print(f"\n✗ NOT VALIDATED: scoring does not discriminate.")

        print(f"\nLOWEST SCORES (most hackable):")
        for s in sorted(scores.values(), key=lambda x: x.composite_score)[:5]:
            tag = " [HACKABLE]" if s.n_false_positives > 0 else ""
            print(f"  {s.instance_id}: {s.composite_score:.4f}, FPR={s.false_positive_rate:.2f}{tag}")

        print(f"\nHIGHEST SCORES (most robust):")
        for s in sorted(scores.values(), key=lambda x: -x.composite_score)[:5]:
            tag = " [HACKABLE]" if s.n_false_positives > 0 else ""
            print(f"  {s.instance_id}: {s.composite_score:.4f}, FPR={s.false_positive_rate:.2f}{tag}")


def _print_statistical_analysis(sep):
    """Print statistical significance results."""
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    sig_str = "YES (p < 0.05)" if sep.is_significant else "NO (p >= 0.05)"
    print(f"\n  Mann-Whitney U = {sep.mann_whitney_u:.1f}")
    print(f"  p-value = {sep.p_value:.6f}")
    print(f"  Significant: {sig_str}")
    print(f"  Effect size (rank-biserial r) = {sep.effect_size_r:.4f}")
    print(f"\n  95% CI hackable mean:     [{sep.ci_lower_hackable:.4f}, {sep.ci_upper_hackable:.4f}]")
    print(f"  95% CI non-hackable mean: [{sep.ci_lower_non_hackable:.4f}, {sep.ci_upper_non_hackable:.4f}]")
    print(f"\n  Optimal threshold: {sep.optimal_threshold:.4f}")
    print(f"  At threshold — sensitivity: {sep.threshold_sensitivity:.2f}, specificity: {sep.threshold_specificity:.2f}")
    print(f"  Youden's J: {sep.threshold_youden_j:.4f}")


def _print_weight_sensitivity(sensitivity):
    """Print weight sensitivity analysis."""
    print("\n" + "=" * 70)
    print("WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"\n  {'Config':<25} {'Hackable':>10} {'Non-hack':>10} {'Delta':>8} {'p-value':>10}")
    print("  " + "-" * 65)
    for ws in sensitivity:
        sig = "*" if ws.p_value < 0.05 else " "
        print(f"  {ws.weight_config_name:<25} {ws.mean_hackable:>10.4f} {ws.mean_non_hackable:>10.4f} {ws.delta:>+8.4f} {ws.p_value:>9.6f} {sig}")
    print(f"\n  * = significant at p < 0.05")
    robust_count = sum(1 for ws in sensitivity if ws.p_value < 0.05)
    print(f"  Result holds under {robust_count}/{len(sensitivity)} weight configurations")


def _print_category_analysis(categories):
    """Print per-exploit-category success rates."""
    print("\n" + "=" * 70)
    print("EXPLOIT CATEGORY ANALYSIS")
    print("=" * 70)
    print(f"\n  {'Category':<30} {'Attempted':>10} {'Verified':>10} {'Rate':>8}")
    print("  " + "-" * 60)
    for c in categories:
        print(f"  {c.category:<30} {c.total_attempts:>10} {c.docker_verified:>10} {c.success_rate:>7.1%}")
    print("=" * 70)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


if __name__ == "__main__":
    main()
