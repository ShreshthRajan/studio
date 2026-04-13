"""
Step 3: Hybrid Verification — compare test suite verdicts against
semi-formal correctness judgments.

For each candidate patch that has Docker verification data, runs the
semi-formal judge and builds a confusion matrix per task.

Detects:
  - False positives: tests pass but judge says incorrect (HACKABLE)
  - False negatives: tests fail but judge says correct (OVERLY STRICT)

Usage:
  python experiments/run_hybrid_verification.py --tasks 5     # First 5 tasks
  python experiments/run_hybrid_verification.py               # All tasks with Docker data
  python experiments/run_hybrid_verification.py --self-consistency  # 3x sampling
  python experiments/run_hybrid_verification.py --dry-run     # Show plan, no API calls
"""

import json
import sys
import os
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from envaudit.scoring.patch_assembler import (
    assemble_candidates_from_phase1,
    load_gold_patches,
)
from envaudit.scoring.semiformal_judge import (
    judge_patch,
    judge_patch_with_self_consistency,
    JudgmentResult,
    CorrectnessVerdict,
)
from envaudit.scoring.hybrid import compute_hybrid_result, HybridResult
from envaudit.scoring.verifier_scorer import CandidatePatch
from envaudit.llm.claude_client import ClaudeClient
from envaudit.data.swebench import load_swebench_verified

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PHASE1_RESULTS = PROJECT_ROOT / "experiments" / "phase1_results.json"
VERIFICATION_LOGS = PROJECT_ROOT / "phase1_verification" / "verification_output"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "hybrid_verification_results.json"


def main():
    parser = argparse.ArgumentParser(description="Run hybrid verification (Step 3)")
    parser.add_argument("--tasks", type=int, default=None, help="Limit tasks")
    parser.add_argument("--self-consistency", action="store_true",
                        help="Use 3x self-consistency sampling (3x cost, more reliable)")
    parser.add_argument("--skip-gold-judge", action="store_true",
                        help="Don't judge gold patches (saves cost, they're always correct)")
    parser.add_argument("--skip-trivial-judge", action="store_true",
                        help="Don't judge trivial patches (saves cost, they're always incorrect)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    # --- Load data ---
    logger.info("Loading phase1 results...")
    logger.info("Loading gold patches...")
    gold_patches = load_gold_patches()

    # Load task data for problem statements
    all_tasks = load_swebench_verified()
    task_lookup = {t.instance_id: t for t in all_tasks}
    # Support __ format
    for t in all_tasks:
        task_lookup[t.instance_id.replace("/", "__")] = t

    # Assemble candidates
    task_candidates = assemble_candidates_from_phase1(
        str(PHASE1_RESULTS), str(VERIFICATION_LOGS), gold_patches,
    )

    # Filter to tasks with Docker data (verifier_pass is not None on at least one exploit)
    tasks_with_docker = {}
    for iid, candidates in task_candidates.items():
        has_docker = any(
            c.verifier_pass is not None and c.source == "exploit"
            for c in candidates
        )
        if has_docker:
            tasks_with_docker[iid] = candidates

    task_ids = sorted(tasks_with_docker.keys())
    if args.tasks:
        task_ids = task_ids[:args.tasks]

    # Count patches to judge
    n_to_judge = 0
    for iid in task_ids:
        for c in tasks_with_docker[iid]:
            if args.skip_gold_judge and c.source == "gold":
                continue
            if args.skip_trivial_judge and c.source == "trivial":
                continue
            if c.verifier_pass is not None:
                n_to_judge += 1

    logger.info("Tasks: %d, Patches to judge: %d", len(task_ids), n_to_judge)
    est_cost = n_to_judge * 0.015 * (3 if args.self_consistency else 1)
    logger.info("Estimated cost: ~$%.2f", est_cost)

    if args.dry_run:
        print(f"\nDRY RUN: Would judge {n_to_judge} patches across {len(task_ids)} tasks.")
        print(f"Estimated cost: ~${est_cost:.2f}")
        return

    # --- Run judgments ---
    client = ClaudeClient()
    all_results: Dict[str, HybridResult] = {}
    total_cost = 0.0

    for idx, iid in enumerate(task_ids):
        task = task_lookup.get(iid)
        if not task:
            logger.warning("Task %s not found in dataset, skipping", iid)
            continue

        candidates = tasks_with_docker[iid]
        judgments: Dict[str, JudgmentResult] = {}

        # Judge each candidate
        for c in candidates:
            if args.skip_gold_judge and c.source == "gold":
                # Gold is always correct — skip judging to save cost
                judgments[c.patch_id] = JudgmentResult(
                    verdict=CorrectnessVerdict.CORRECT,
                    premises="Gold patch (known correct)",
                    trace="N/A — gold patch by definition",
                    conclusion="The gold patch is the known-correct solution.",
                    confidence=1.0,
                    cost_usd=0.0,
                    latency_seconds=0.0,
                )
                continue

            if args.skip_trivial_judge and c.source == "trivial":
                judgments[c.patch_id] = JudgmentResult(
                    verdict=CorrectnessVerdict.INCORRECT,
                    premises="Empty/trivial patch",
                    trace="N/A — empty patch",
                    conclusion="Empty patch cannot solve any problem.",
                    confidence=1.0,
                    cost_usd=0.0,
                    latency_seconds=0.0,
                )
                continue

            if c.verifier_pass is None:
                continue

            # Run semi-formal judgment
            if args.self_consistency:
                judgment = judge_patch_with_self_consistency(
                    client, task.problem_statement, c.patch_text, task.gold_patch,
                )
            else:
                judgment = judge_patch(
                    client, task.problem_statement, c.patch_text, task.gold_patch,
                )

            judgments[c.patch_id] = judgment
            total_cost += judgment.cost_usd

        # Compute hybrid result
        hybrid = compute_hybrid_result(iid, candidates, judgments)
        all_results[iid] = hybrid

        fp_patches = [e.patch_id for e in hybrid.entries if e.category == "FP"]
        fn_patches = [e.patch_id for e in hybrid.entries if e.category == "FN"]
        logger.info("[%d/%d] %s: P=%.2f R=%.2f F1=%.2f | FP=%d FN=%d | cost=$%.4f",
                    idx + 1, len(task_ids), iid,
                    hybrid.precision, hybrid.recall, hybrid.f1,
                    hybrid.fp, hybrid.fn, hybrid.total_judge_cost)
        if fp_patches:
            logger.info("  HACKABLE patches: %s", fp_patches)
        if fn_patches:
            logger.info("  OVERLY STRICT on: %s", fn_patches)

    # --- Print summary ---
    _print_summary(all_results, total_cost)

    # --- Save ---
    output = {
        "total_tasks": len(all_results),
        "total_cost_usd": round(total_cost, 4),
        "aggregate": _compute_aggregate(all_results),
        "per_task": {
            iid: {
                "tp": r.tp, "fp": r.fp, "fn": r.fn, "tn": r.tn,
                "precision": r.precision, "recall": r.recall, "f1": r.f1,
                "false_positive_rate": r.false_positive_rate,
                "false_negative_rate": r.false_negative_rate,
                "n_judged": r.n_judged,
                "entries": [
                    {"patch_id": e.patch_id, "verifier_pass": e.verifier_pass,
                     "judge_correct": e.judge_correct, "category": e.category}
                    for e in r.entries
                ],
            }
            for iid, r in sorted(all_results.items())
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved to %s", args.output)


def _compute_aggregate(results: Dict[str, HybridResult]) -> Dict:
    """Compute aggregate confusion matrix across all tasks."""
    total_tp = sum(r.tp for r in results.values())
    total_fp = sum(r.fp for r in results.values())
    total_fn = sum(r.fn for r in results.values())
    total_tn = sum(r.tn for r in results.values())

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        "tp": total_tp, "fp": total_fp, "fn": total_fn, "tn": total_tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(total_fp / max(total_fp + total_tn, 1), 4),
        "false_negative_rate": round(total_fn / max(total_fn + total_tp, 1), 4),
        "tasks_with_fp": sum(1 for r in results.values() if r.fp > 0),
        "tasks_with_fn": sum(1 for r in results.values() if r.fn > 0),
    }


def _print_summary(results: Dict[str, HybridResult], total_cost: float):
    """Print hybrid verification summary."""
    agg = _compute_aggregate(results)

    print("\n" + "=" * 70)
    print("HYBRID VERIFICATION RESULTS")
    print("=" * 70)
    print(f"\n  Tasks judged: {len(results)}")
    print(f"  Total cost: ${total_cost:.4f}")

    print(f"\n  AGGREGATE CONFUSION MATRIX:")
    print(f"                     Judge: Correct    Judge: Incorrect")
    print(f"  Tests Pass (✓):    TP = {agg['tp']:>4}          FP = {agg['fp']:>4}  ← HACKABLE")
    print(f"  Tests Fail (✗):    FN = {agg['fn']:>4}          TN = {agg['tn']:>4}")
    print(f"                     ↑ OVERLY STRICT")

    print(f"\n  Precision: {agg['precision']:.4f} (of patches that pass tests, how many are correct)")
    print(f"  Recall:    {agg['recall']:.4f} (of correct patches, how many pass tests)")
    print(f"  F1:        {agg['f1']:.4f}")
    print(f"\n  False Positive Rate: {agg['false_positive_rate']:.4f} (hackability)")
    print(f"  False Negative Rate: {agg['false_negative_rate']:.4f} (over-strictness)")

    print(f"\n  Tasks with false positives (hackable): {agg['tasks_with_fp']}")
    print(f"  Tasks with false negatives (overly strict): {agg['tasks_with_fn']}")

    # Show tasks with highest FP
    fp_tasks = [(iid, r) for iid, r in results.items() if r.fp > 0]
    if fp_tasks:
        print(f"\n  HACKABLE TASKS (tests pass incorrect patches):")
        for iid, r in sorted(fp_tasks, key=lambda x: -x[1].fp):
            print(f"    {iid}: {r.fp} false positives")

    fn_tasks = [(iid, r) for iid, r in results.items() if r.fn > 0]
    if fn_tasks:
        print(f"\n  OVERLY STRICT TASKS (tests reject correct patches):")
        for iid, r in sorted(fn_tasks, key=lambda x: -x[1].fn):
            print(f"    {iid}: {r.fn} false negatives")

    print("=" * 70)


if __name__ == "__main__":
    main()
