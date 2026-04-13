"""
Steps 4+5: Compute composite Environment Quality Score (EQS) for all tasks.

Combines:
  - Step 1: Verifier scoring (experiments/verifier_scores.json)
  - Step 2: Exploit success rates (phase1_verification/verification_output/)
  - Step 3: Hybrid verification F1 (experiments/hybrid_verification_results.json)
  - Step 4: Difficulty profiling (SWE-bench/experiments solve rates)

Outputs:
  - experiments/composite_scores.json — per-task EQS with verdicts
  - experiments/difficulty_cache.json — cached solve rates from GitHub

Usage:
  python experiments/run_composite_scoring.py                  # Full run
  python experiments/run_composite_scoring.py --skip-difficulty # Skip GitHub fetch
  python experiments/run_composite_scoring.py --difficulty-cache experiments/difficulty_cache.json
"""

import json
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from envaudit.scoring.verifier_scorer import VerifierScore
from envaudit.scoring.hybrid import HybridResult
from envaudit.scoring.difficulty import (
    DifficultyProfile,
    profile_difficulty,
    fetch_swebench_solve_rates,
    load_solve_rates_from_results,
)
from envaudit.scoring.composite import compute_eqs, CompositeResult
from envaudit.scoring.patch_assembler import _parse_docker_logs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
VERIFIER_SCORES = PROJECT_ROOT / "experiments" / "verifier_scores.json"
HYBRID_RESULTS = PROJECT_ROOT / "experiments" / "hybrid_verification_results.json"
VERIFICATION_LOGS = PROJECT_ROOT / "phase1_verification" / "verification_output"
PHASE1_RESULTS = PROJECT_ROOT / "experiments" / "phase1_results.json"
DIFFICULTY_CACHE = PROJECT_ROOT / "experiments" / "difficulty_cache.json"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "composite_scores.json"


def main():
    parser = argparse.ArgumentParser(description="Compute composite EQS (Steps 4+5)")
    parser.add_argument("--skip-difficulty", action="store_true",
                        help="Skip difficulty profiling (no GitHub fetch)")
    parser.add_argument("--difficulty-cache", type=str, default=str(DIFFICULTY_CACHE),
                        help="Path to difficulty cache file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    # --- Load Step 1: Verifier Scores ---
    verifier_scores: Dict[str, VerifierScore] = {}
    if VERIFIER_SCORES.exists():
        with open(VERIFIER_SCORES) as f:
            vs_data = json.load(f)
        for iid, task_data in vs_data.get("per_task", {}).items():
            verifier_scores[iid] = VerifierScore(
                instance_id=iid,
                top1_accuracy=task_data["top1_accuracy"],
                bottom1_accuracy=task_data["bottom1_accuracy"],
                spearman_rho=task_data["spearman_rho"],
                mae=task_data["mae"],
                composite_score=task_data["composite_score"],
                n_candidates=task_data["n_candidates"],
                n_false_positives=task_data["n_false_positives"],
                n_false_negatives=task_data["n_false_negatives"],
                false_positive_rate=task_data["false_positive_rate"],
                false_negative_rate=task_data["false_negative_rate"],
            )
        logger.info("Loaded verifier scores for %d tasks", len(verifier_scores))
    else:
        logger.warning("No verifier scores found at %s", VERIFIER_SCORES)

    # --- Load Step 2: Exploit success rates ---
    exploit_rates: Dict[str, float] = {}
    docker_results = _parse_docker_logs(str(VERIFICATION_LOGS))
    with open(PHASE1_RESULTS) as f:
        phase1_data = json.load(f)

    for task in phase1_data:
        iid = task["instance_id"]
        docker_key = iid.replace("/", "__")
        attempted = 0
        verified = 0
        for idx in range(len(task.get("exploit_patches", []))):
            key = f"{docker_key}_exploit_{idx}"
            if key in docker_results:
                attempted += 1
                if docker_results[key]:
                    verified += 1
        if attempted > 0:
            exploit_rates[iid] = verified / attempted
    logger.info("Loaded exploit rates for %d tasks", len(exploit_rates))

    # --- Load Step 3: Hybrid Verification ---
    hybrid_results: Dict[str, HybridResult] = {}
    if HYBRID_RESULTS.exists():
        with open(HYBRID_RESULTS) as f:
            hv_data = json.load(f)
        for iid, task_data in hv_data.get("per_task", {}).items():
            hybrid_results[iid] = HybridResult(
                instance_id=iid,
                entries=[],  # Don't need full entries for composite scoring
                tp=task_data["tp"], fp=task_data["fp"],
                fn=task_data["fn"], tn=task_data["tn"],
                precision=task_data["precision"],
                recall=task_data["recall"],
                f1=task_data["f1"],
                false_positive_rate=task_data["false_positive_rate"],
                false_negative_rate=task_data["false_negative_rate"],
                n_judged=task_data["n_judged"],
                n_skipped=0,
                total_judge_cost=0.0,
            )
        logger.info("Loaded hybrid results for %d tasks", len(hybrid_results))
    else:
        logger.warning("No hybrid results found at %s", HYBRID_RESULTS)

    # --- Step 4: Difficulty Profiling ---
    difficulty_profiles: Dict[str, DifficultyProfile] = {}
    if not args.skip_difficulty:
        # Get the set of instance_ids we care about
        all_iids = set(verifier_scores.keys()) | set(exploit_rates.keys()) | set(hybrid_results.keys())

        solve_rates = fetch_swebench_solve_rates(
            cache_path=args.difficulty_cache,
            target_instance_ids=all_iids if all_iids else None,
        )

        if solve_rates:
            for iid, rate in solve_rates.items():
                difficulty_profiles[iid] = profile_difficulty(iid, rate, n_models=len(solve_rates))
            logger.info("Computed difficulty profiles for %d tasks", len(difficulty_profiles))

            # Print difficulty distribution
            tier_counts = Counter(d.difficulty_tier for d in difficulty_profiles.values())
            logger.info("Difficulty distribution: %s", dict(tier_counts))
        else:
            logger.warning("No solve rates available — skipping difficulty dimension")
    else:
        logger.info("Skipping difficulty profiling (--skip-difficulty)")

    # --- Step 5: Compute Composite EQS ---
    all_iids = sorted(set(verifier_scores.keys()) | set(exploit_rates.keys())
                      | set(hybrid_results.keys()) | set(difficulty_profiles.keys()))

    results: Dict[str, CompositeResult] = {}
    for iid in all_iids:
        result = compute_eqs(
            instance_id=iid,
            verifier=verifier_scores.get(iid),
            hybrid=hybrid_results.get(iid),
            difficulty=difficulty_profiles.get(iid),
            exploit_success_rate=exploit_rates.get(iid),
        )
        results[iid] = result

    logger.info("Computed EQS for %d tasks", len(results))

    # --- Print Summary ---
    _print_summary(results)

    # --- Save ---
    output = {
        "total_tasks": len(results),
        "verdict_counts": dict(Counter(r.verdict for r in results.values())),
        "mean_eqs": round(sum(r.eqs for r in results.values()) / max(len(results), 1), 4),
        "dimensions_available": {
            "verifier": len(verifier_scores),
            "exploit": len(exploit_rates),
            "hybrid": len(hybrid_results),
            "difficulty": len(difficulty_profiles),
        },
        "per_task": {
            iid: {
                "eqs": r.eqs,
                "verdict": r.verdict,
                "dimension_scores": r.dimension_scores,
                "weaknesses": [
                    {
                        "dimension": w.dimension,
                        "severity": w.severity,
                        "description": w.description,
                        "fixable": w.fixable,
                        "action": w.action,
                    }
                    for w in r.weaknesses
                ],
            }
            for iid, r in sorted(results.items())
        },
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved composite scores to %s", output_path)


def _print_summary(results: Dict[str, CompositeResult]):
    """Print comprehensive summary."""
    verdicts = Counter(r.verdict for r in results.values())
    mean_eqs = sum(r.eqs for r in results.values()) / max(len(results), 1)

    print("\n" + "=" * 70)
    print("COMPOSITE ENVIRONMENT QUALITY SCORES (EQS)")
    print("=" * 70)

    print(f"\n  Tasks scored: {len(results)}")
    print(f"  Mean EQS: {mean_eqs:.4f}")
    print(f"\n  Verdicts:")
    print(f"    KEEP:  {verdicts.get('KEEP', 0):>3} ({verdicts.get('KEEP', 0)/max(len(results),1):.0%})")
    print(f"    FIX:   {verdicts.get('FIX', 0):>3} ({verdicts.get('FIX', 0)/max(len(results),1):.0%})")
    print(f"    DROP:  {verdicts.get('DROP', 0):>3} ({verdicts.get('DROP', 0)/max(len(results),1):.0%})")

    # Per-dimension means
    dim_totals: Dict[str, list] = {}
    for r in results.values():
        for dim, score in r.dimension_scores.items():
            dim_totals.setdefault(dim, []).append(score)

    if dim_totals:
        print(f"\n  Per-dimension means:")
        for dim in ["verifier", "exploit", "hybrid", "difficulty"]:
            if dim in dim_totals:
                scores = dim_totals[dim]
                print(f"    {dim:<12}: {sum(scores)/len(scores):.4f} (n={len(scores)})")

    # Weakness analysis
    all_weaknesses = [w for r in results.values() for w in r.weaknesses]
    weakness_counts = Counter(w.dimension for w in all_weaknesses)
    severity_counts = Counter(w.severity for w in all_weaknesses)

    if all_weaknesses:
        print(f"\n  Weaknesses found: {len(all_weaknesses)}")
        print(f"    By dimension: {dict(weakness_counts)}")
        print(f"    By severity:  {dict(severity_counts)}")
        fixable = sum(1 for w in all_weaknesses if w.fixable)
        print(f"    Fixable: {fixable}/{len(all_weaknesses)}")

    # Show worst tasks
    print(f"\n  LOWEST EQS (worst environments):")
    for r in sorted(results.values(), key=lambda x: x.eqs)[:5]:
        dims = ", ".join(f"{d}={s:.2f}" for d, s in r.dimension_scores.items())
        print(f"    {r.instance_id}: EQS={r.eqs:.4f} [{r.verdict}] ({dims})")

    print(f"\n  HIGHEST EQS (best environments):")
    for r in sorted(results.values(), key=lambda x: -x.eqs)[:5]:
        dims = ", ".join(f"{d}={s:.2f}" for d, s in r.dimension_scores.items())
        print(f"    {r.instance_id}: EQS={r.eqs:.4f} [{r.verdict}] ({dims})")

    print("=" * 70)


if __name__ == "__main__":
    main()
