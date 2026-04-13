"""
Step 6: Run the optimization loop — the "hill climbing."

For each FIX task: generate targeted test augmentations, estimate improvement.

Usage:
  python experiments/run_optimization.py                # All FIX tasks
  python experiments/run_optimization.py --tasks 3      # First 3 FIX tasks
  python experiments/run_optimization.py --dry-run      # Show plan only
  python experiments/run_optimization.py --max-iter 2   # Max 2 iterations
"""

import json
import sys
import os
import logging
import argparse
import glob
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from envaudit.optimization.augmenter import TestAugmenter
from envaudit.optimization.loop import run_optimization_loop, OptimizationResult
from envaudit.scoring.composite import CompositeResult
from envaudit.llm.claude_client import ClaudeClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
COMPOSITE_SCORES = PROJECT_ROOT / "experiments" / "composite_scores.json"
PHASE1_RESULTS = PROJECT_ROOT / "experiments" / "phase1_results.json"
VERIFICATION_LOGS = PROJECT_ROOT / "phase1_verification" / "verification_output"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "optimization_results.json"


def main():
    parser = argparse.ArgumentParser(description="Run optimization loop (Step 6)")
    parser.add_argument("--tasks", type=int, default=None, help="Limit FIX tasks")
    parser.add_argument("--max-iter", type=int, default=3, help="Max optimization iterations")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    # --- Load composite scores to find FIX tasks ---
    with open(COMPOSITE_SCORES) as f:
        cs = json.load(f)

    fix_tasks = {}
    for iid, task in cs["per_task"].items():
        if task["verdict"] == "FIX":
            fix_tasks[iid] = CompositeResult(
                instance_id=iid,
                eqs=task["eqs"],
                verdict=task["verdict"],
                dimension_scores=task["dimension_scores"],
                weaknesses=[],  # Not needed for the loop
                weights={},
            )

    if args.tasks:
        fix_ids = sorted(fix_tasks.keys())[:args.tasks]
        fix_tasks = {iid: fix_tasks[iid] for iid in fix_ids}

    logger.info("FIX tasks to optimize: %d", len(fix_tasks))

    # --- Load task data (problem statements, patches, exploits) ---
    with open(PHASE1_RESULTS) as f:
        phase1 = json.load(f)
    phase1_lookup = {t["instance_id"]: t for t in phase1}

    # Load SWE-bench task data
    from envaudit.data.swebench import load_swebench_verified
    all_tasks = load_swebench_verified()
    swe_lookup = {t.instance_id: t for t in all_tasks}
    for t in all_tasks:
        swe_lookup[t.instance_id.replace("/", "__")] = t

    # Parse Docker results to find which exploits passed
    docker_passed = _get_passed_exploits()

    # Build task data dict for the loop
    task_data: Dict[str, Dict] = {}
    for iid in fix_tasks:
        swe_task = swe_lookup.get(iid)
        p1_task = phase1_lookup.get(iid)
        if not swe_task or not p1_task:
            continue

        passed = docker_passed.get(iid, [])
        passed_exploits = []
        for idx in passed:
            strategies = p1_task.get("strategies", [])
            patches = p1_task.get("exploit_patches", [])
            passed_exploits.append({
                "patch_id": f"{iid}_exploit_{idx}",
                "patch_text": patches[idx] if idx < len(patches) else "",
                "strategy_name": strategies[idx]["name"] if idx < len(strategies) else "unknown",
                "strategy_description": strategies[idx].get("description", "") if idx < len(strategies) else "",
            })

        task_data[iid] = {
            "problem_statement": swe_task.problem_statement,
            "test_patch": swe_task.test_patch,
            "gold_patch": swe_task.gold_patch,
            "passed_exploits": passed_exploits,
        }

    if args.dry_run:
        _print_dry_run(fix_tasks, task_data)
        return

    # --- Run optimization loop ---
    client = ClaudeClient()
    augmenter = TestAugmenter(client)

    result = run_optimization_loop(
        fix_tasks=fix_tasks,
        task_data=task_data,
        augmenter=augmenter,
        max_iterations=args.max_iter,
    )

    # --- Print summary ---
    _print_summary(result)

    # --- Save ---
    output = {
        "total_iterations": result.total_iterations,
        "converged": result.converged,
        "initial_mean_eqs": result.initial_mean_eqs,
        "final_mean_eqs": result.final_mean_eqs,
        "total_improvement": result.total_improvement,
        "total_cost_usd": result.total_cost_usd,
        "tasks_upgraded_total": result.tasks_upgraded_total,
        "convergence_curve": result.convergence_curve,
        "iterations": [
            {
                "iteration": it.iteration,
                "tasks_targeted": it.tasks_targeted,
                "augmentations_generated": it.augmentations_generated,
                "tasks_improved": it.tasks_improved,
                "tasks_upgraded": it.tasks_upgraded,
                "mean_eqs_before": it.mean_eqs_before,
                "mean_eqs_after": it.mean_eqs_after,
                "improvement": it.improvement,
                "cost_usd": it.cost_usd,
                "per_task": it.per_task,
            }
            for it in result.iterations
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved to %s", args.output)


def _get_passed_exploits() -> Dict[str, List[int]]:
    """Get which exploit indices passed Docker for each task."""
    passed = {}
    for log_path in sorted(glob.glob(str(VERIFICATION_LOGS / "*.log"))):
        with open(log_path) as f:
            content = f.read()
        if "Instances resolved: 1" in content:
            basename = os.path.basename(log_path).replace(".log", "")
            parts = basename.rsplit("_exploit_", 1)
            if len(parts) == 2:
                task_key = parts[0]
                idx = int(parts[1])
                passed.setdefault(task_key, []).append(idx)
    return passed


def _print_dry_run(fix_tasks, task_data):
    print("\n" + "=" * 70)
    print("OPTIMIZATION LOOP — DRY RUN")
    print("=" * 70)
    for iid in sorted(fix_tasks.keys()):
        data = task_data.get(iid, {})
        n_exploits = len(data.get("passed_exploits", []))
        print(f"  {iid}: EQS={fix_tasks[iid].eqs:.4f}, {n_exploits} passed exploits to block")
        for exp in data.get("passed_exploits", []):
            print(f"    - {exp['strategy_name']}: {exp['strategy_description'][:80]}")
    est_cost = len(fix_tasks) * 0.05 * 3  # ~$0.05/task * 3 iterations
    print(f"\nEstimated cost: ~${est_cost:.2f}")
    print("=" * 70)


def _print_summary(result: OptimizationResult):
    print("\n" + "=" * 70)
    print("OPTIMIZATION LOOP RESULTS")
    print("=" * 70)

    print(f"\n  Iterations: {result.total_iterations}")
    print(f"  Converged: {result.converged}")
    print(f"  Initial mean EQS: {result.initial_mean_eqs:.4f}")
    print(f"  Final mean EQS:   {result.final_mean_eqs:.4f}")
    print(f"  Total improvement: {result.total_improvement:+.4f}")
    print(f"  Tasks upgraded (FIX → KEEP): {result.tasks_upgraded_total}")
    print(f"  Total cost: ${result.total_cost_usd:.4f}")

    print(f"\n  Convergence curve: {' → '.join(f'{x:.4f}' for x in result.convergence_curve)}")

    for it in result.iterations:
        print(f"\n  --- Iteration {it.iteration} ---")
        print(f"    Tasks targeted: {it.tasks_targeted}")
        print(f"    Augmentations generated: {it.augmentations_generated}")
        print(f"    Tasks improved: {it.tasks_improved}/{it.tasks_targeted}")
        print(f"    Tasks upgraded to KEEP: {it.tasks_upgraded}")
        print(f"    EQS: {it.mean_eqs_before:.4f} → {it.mean_eqs_after:.4f} (+{it.improvement:.4f})")

        for td in it.per_task:
            status = "UPGRADED" if td["upgraded"] else ("improved" if td["improved"] else "unchanged")
            print(f"      {td['instance_id']}: {td['eqs_before']:.4f} → {td['eqs_after']:.4f} [{status}] ({td['n_augmentations']} augs)")

    print("=" * 70)


if __name__ == "__main__":
    main()
