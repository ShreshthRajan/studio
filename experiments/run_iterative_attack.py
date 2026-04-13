"""
Step 2: Run iterative adversarial attack (Rounds 2+3) on tasks
that have Round 1 results and Docker verification data.

Uses Round 1 failure information to generate targeted exploits (Round 2)
and EvolveCoder-style "almost correct" patches (Round 3).

Inputs:
  - experiments/phase1_results.json (Round 1 strategies)
  - phase1_verification/verification_output/*.log (Round 1 Docker results)
  - SWE-bench Verified dataset (task data)

Outputs:
  - experiments/iterative_attack_results.json (Rounds 2+3 strategies + patches)
  - experiments/verify_inputs_r2r3/ (prediction JSONLs for Docker verification)

Usage:
  python experiments/run_iterative_attack.py                    # All 49 tasks with Docker data
  python experiments/run_iterative_attack.py --tasks 10         # First 10 tasks
  python experiments/run_iterative_attack.py --only-failed      # Only tasks where Round 1 exploits all failed
  python experiments/run_iterative_attack.py --dry-run          # Show what would run without calling Claude
"""

import json
import sys
import os
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from envaudit.agents.iterative_attacker import IterativeAttacker, IterativeAttackResult
from envaudit.agents.base import TaskData
from envaudit.scoring.patch_assembler import _parse_docker_logs
from envaudit.llm.claude_client import ClaudeClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PHASE1_RESULTS = PROJECT_ROOT / "experiments" / "phase1_results.json"
VERIFICATION_LOGS = PROJECT_ROOT / "phase1_verification" / "verification_output"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "iterative_attack_results.json"
VERIFY_INPUTS_DIR = PROJECT_ROOT / "experiments" / "verify_inputs_r2r3"


def main():
    parser = argparse.ArgumentParser(description="Run iterative adversarial attack (Rounds 2+3)")
    parser.add_argument("--tasks", type=int, default=None, help="Limit number of tasks to process")
    parser.add_argument("--only-failed", action="store_true",
                        help="Only attack tasks where ALL Round 1 exploits failed Docker")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without calling Claude API")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    # --- Load Round 1 data ---
    with open(PHASE1_RESULTS) as f:
        phase1_data = json.load(f)
    logger.info("Loaded %d tasks from Round 1", len(phase1_data))

    # --- Parse Docker verification logs ---
    docker_results = _parse_docker_logs(str(VERIFICATION_LOGS))
    logger.info("Parsed %d Docker verification logs", len(docker_results))

    # --- Load SWE-bench task data ---
    logger.info("Loading SWE-bench Verified tasks...")
    from envaudit.data.swebench import load_swebench_verified
    all_tasks = load_swebench_verified()
    # SWE-bench uses "/" format (e.g., "astropy/astropy-12907")
    # phase1_results uses "__" format (e.g., "astropy__astropy-12907")
    # Build lookup supporting both formats
    task_lookup = {}
    for t in all_tasks:
        task_lookup[t.instance_id] = t
        task_lookup[t.instance_id.replace("/", "__")] = t
    logger.info("Loaded %d task definitions", len(all_tasks))

    # --- Build per-task Docker results ---
    task_docker: Dict[str, Dict[int, bool]] = {}
    for log_key, passed in docker_results.items():
        parts = log_key.rsplit("_exploit_", 1)
        if len(parts) == 2:
            task_key = parts[0]  # e.g., "astropy__astropy-12907"
            exploit_idx = int(parts[1])
            if task_key not in task_docker:
                task_docker[task_key] = {}
            task_docker[task_key][exploit_idx] = passed

    # --- Determine which tasks to attack ---
    tasks_to_attack = []
    for task_data in phase1_data:
        iid = task_data["instance_id"]
        if iid not in task_lookup:
            continue

        docker_res = task_docker.get(iid, {})

        if args.only_failed:
            # Skip tasks where any exploit already passed
            if any(docker_res.values()):
                continue

        tasks_to_attack.append((task_data, docker_res))

    if args.tasks:
        tasks_to_attack = tasks_to_attack[:args.tasks]

    logger.info("Tasks to attack: %d", len(tasks_to_attack))

    if args.dry_run:
        _print_dry_run(tasks_to_attack)
        return

    # --- Run iterative attacks ---
    client = ClaudeClient()
    attacker = IterativeAttacker(client)

    results: List[Dict] = []
    total_cost = 0.0
    total_new_strategies = 0

    for i, (task_data, docker_res) in enumerate(tasks_to_attack):
        iid = task_data["instance_id"]
        task = task_lookup[iid]
        round1_strategies = task_data.get("strategies", [])

        logger.info("[%d/%d] Attacking %s (Round 1: %d strategies, Docker: %s)",
                    i + 1, len(tasks_to_attack), iid,
                    len(round1_strategies),
                    f"{sum(docker_res.values())}/{len(docker_res)} passed" if docker_res else "no data")

        t0 = time.time()
        attack_result = attacker.run_all_rounds(
            task=task,
            round1_strategies=round1_strategies,
            round1_docker_results=docker_res if docker_res else None,
        )
        elapsed = time.time() - t0

        total_cost += attack_result.total_cost_usd
        total_new_strategies += attack_result.total_strategies

        results.append({
            "instance_id": iid,
            "rounds": [
                {
                    "round_num": r.round_num,
                    "n_strategies": len(r.strategies),
                    "strategies": [
                        {
                            "name": s.name,
                            "category": s.category,
                            "confidence": s.confidence,
                            "description": s.description,
                            "reasoning": s.reasoning,
                        }
                        for s in r.strategies
                    ],
                    "cost_usd": r.cost_usd,
                }
                for r in attack_result.rounds
            ],
            "exploit_patches": attack_result.all_exploit_patches,
            "total_strategies": attack_result.total_strategies,
            "max_confidence": attack_result.max_confidence,
            "total_cost_usd": attack_result.total_cost_usd,
            "elapsed_seconds": round(elapsed, 2),
        })

        logger.info("  → %d new strategies (max conf: %d), cost: $%.4f, time: %.1fs",
                    attack_result.total_strategies, attack_result.max_confidence,
                    attack_result.total_cost_usd, elapsed)

    # --- Save results ---
    output = {
        "total_tasks": len(results),
        "total_new_strategies": total_new_strategies,
        "total_cost_usd": round(total_cost, 4),
        "per_task": results,
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved iterative attack results to %s", output_path)

    # --- Prepare verification inputs ---
    _prepare_verification_inputs(results)

    # --- Print summary ---
    _print_summary(results, total_cost)


def _prepare_verification_inputs(results: List[Dict]):
    """Write prediction JSONLs for Docker verification of new exploits."""
    VERIFY_INPUTS_DIR.mkdir(exist_ok=True)

    n_files = 0
    for task_result in results:
        iid = task_result["instance_id"]
        # Convert instance_id format for swebench harness
        swebench_id = iid.replace("/", "__")

        for idx, patch in enumerate(task_result["exploit_patches"]):
            if not patch.strip():
                continue

            pred = {
                "instance_id": iid,
                "model_name_or_path": f"envaudit_r2r3_{swebench_id}_exploit_{idx}",
                "model_patch": patch,
            }

            out_file = VERIFY_INPUTS_DIR / f"{swebench_id}_r2r3_exploit_{idx}.jsonl"
            with open(out_file, "w") as f:
                f.write(json.dumps(pred) + "\n")
            n_files += 1

    logger.info("Wrote %d verification input files to %s", n_files, VERIFY_INPUTS_DIR)


def _print_dry_run(tasks_to_attack):
    """Show what would be attacked without calling Claude."""
    print("\n" + "=" * 70)
    print("DRY RUN — Would attack these tasks:")
    print("=" * 70)

    n_all_failed = 0
    n_some_passed = 0
    n_no_docker = 0

    for task_data, docker_res in tasks_to_attack:
        iid = task_data["instance_id"]
        if not docker_res:
            status = "no Docker data"
            n_no_docker += 1
        elif any(docker_res.values()):
            passed = sum(docker_res.values())
            status = f"{passed}/{len(docker_res)} exploits passed Docker"
            n_some_passed += 1
        else:
            status = f"0/{len(docker_res)} exploits passed Docker"
            n_all_failed += 1
        print(f"  {iid}: {status}")

    print(f"\nSummary: {len(tasks_to_attack)} tasks")
    print(f"  All exploits failed: {n_all_failed} (Round 2 most valuable here)")
    print(f"  Some exploits passed: {n_some_passed} (already hackable, Round 3 adds depth)")
    print(f"  No Docker data: {n_no_docker}")
    est_cost = len(tasks_to_attack) * 0.06  # ~$0.03/round × 2 rounds
    print(f"\nEstimated cost: ~${est_cost:.2f}")
    print("=" * 70)


def _print_summary(results: List[Dict], total_cost: float):
    """Print attack summary."""
    print("\n" + "=" * 70)
    print("ITERATIVE ATTACK SUMMARY (Rounds 2+3)")
    print("=" * 70)

    total_strategies = sum(r["total_strategies"] for r in results)
    total_patches = sum(len(r["exploit_patches"]) for r in results)
    max_confs = [r["max_confidence"] for r in results if r["max_confidence"] > 0]
    high_conf_tasks = sum(1 for r in results if r["max_confidence"] >= 7)

    print(f"\n  Tasks attacked: {len(results)}")
    print(f"  Total new strategies: {total_strategies}")
    print(f"  Total new exploit patches: {total_patches}")
    print(f"  Tasks with high-confidence (>=7) new exploits: {high_conf_tasks}")
    if max_confs:
        print(f"  Mean max confidence: {sum(max_confs)/len(max_confs):.1f}/10")
    print(f"  Total cost: ${total_cost:.4f}")

    # Per-round breakdown
    r2_strategies = sum(
        sum(len(rd["strategies"]) for rd in r["rounds"] if rd["round_num"] == 2)
        for r in results
    )
    r3_strategies = sum(
        sum(len(rd["strategies"]) for rd in r["rounds"] if rd["round_num"] == 3)
        for r in results
    )
    print(f"\n  Round 2 (failure-informed): {r2_strategies} strategies")
    print(f"  Round 3 (almost-correct): {r3_strategies} strategies")

    print(f"\n  Verification inputs written to: {VERIFY_INPUTS_DIR}")
    print(f"  → Push to GitHub and run GH Actions to Docker-verify new exploits")
    print("=" * 70)


if __name__ == "__main__":
    main()
