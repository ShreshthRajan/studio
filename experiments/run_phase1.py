#!/usr/bin/env python3
"""
Phase 1: Proof of Concept — Hackability Attacker on 10 SWE-bench Verified tasks.

This is the go/no-go gate. We:
1. Load 10 SWE-bench Verified tasks
2. Run the Hackability Attacker (pattern + LLM) on each
3. For each high-confidence exploit, attempt Docker verification
4. Report results and determine BANGER / STRONG / CHECK / DEAD

Usage:
    # Pattern-only (no API calls, no Docker):
    python3 experiments/run_phase1.py --pattern-only

    # Full (pattern + Claude LLM):
    python3 experiments/run_phase1.py

    # Full + Docker verification:
    python3 experiments/run_phase1.py --verify

Run from project root: cd ~/projects/studio
"""

import asyncio
import argparse
import json
import sys
import os
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envaudit.data.swebench import load_swebench_verified, extract_test_code_from_diff
from envaudit.agents.hackability import HackabilityAttacker
from envaudit.agents.base import TaskData
from envaudit.llm.claude_client import ClaudeClient
from envaudit.docker.verifier import is_docker_available, verify_exploit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_task_summary(task: TaskData):
    """Print a brief summary of a task."""
    test_code = extract_test_code_from_diff(task.test_patch)
    print(f"  ID: {task.instance_id}")
    print(f"  Repo: {task.repo}")
    print(f"  Problem: {task.problem_statement[:120]}...")
    print(f"  FAIL_TO_PASS: {len(task.fail_to_pass)} tests")
    print(f"  PASS_TO_PASS: {len(task.pass_to_pass)} tests")
    print(f"  Test code: {len(test_code.split(chr(10)))} lines")
    print(f"  Gold patch: {len(task.gold_patch.split(chr(10)))} lines")


async def run_phase1(n_tasks: int = 10, pattern_only: bool = False,
                     do_verify: bool = False):
    """Run Phase 1 proof of concept."""

    print_header("PHASE 1: Proof of Concept — Hackability Attacker")
    print(f"Config: n_tasks={n_tasks}, pattern_only={pattern_only}, docker_verify={do_verify}")

    # Step 1: Load data
    print_header("Step 1: Loading SWE-bench Verified data")
    tasks = load_swebench_verified(limit=n_tasks)
    print(f"Loaded {len(tasks)} tasks from SWE-bench Verified")
    print()

    for i, t in enumerate(tasks[:3]):
        print(f"--- Sample task {i+1} ---")
        print_task_summary(t)
        print()

    # Step 2: Initialize attacker
    claude = None
    if not pattern_only:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            # Try loading from multiver .env
            env_path = Path.home() / "projects" / "multiver" / ".env"
            if env_path.exists():
                for line in env_path.read_text().split("\n"):
                    if line.startswith("ANTHROPIC_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        os.environ["ANTHROPIC_API_KEY"] = api_key
                        break

        if api_key:
            claude = ClaudeClient(model="claude-sonnet-4-20250514", api_key=api_key)
            print("Claude client initialized (Sonnet 4.6)")
        else:
            print("WARNING: No ANTHROPIC_API_KEY found. Running pattern-only mode.")
            pattern_only = True

    attacker = HackabilityAttacker(claude_client=claude)

    # Step 3: Run on tasks
    print_header("Step 2: Running Hackability Attacker")
    results = []
    total_cost = 0.0

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Analyzing: {task.instance_id}")
        t0 = time.time()

        result = await attacker.analyze(task)
        elapsed = time.time() - t0
        total_cost += result.cost_usd

        n_strategies = result.metrics.get("num_strategies", 0)
        max_conf = result.metrics.get("max_exploit_confidence", 0)

        print(f"  Verdict: {result.verdict} | Score: {result.score:.2f} | "
              f"Confidence: {result.confidence:.2f}")
        print(f"  Pattern flags: {len(result.metrics.get('pattern_flags', []))}")
        print(f"  LLM strategies: {n_strategies} | Max confidence: {max_conf}")
        print(f"  Time: {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}")

        # Print exploit strategies
        for s in result.metrics.get("strategies", []):
            emoji = "!!!" if s["confidence"] >= 7 else "  "
            print(f"  {emoji} [{s['confidence']}/10] {s['name']}: {s['description'][:80]}...")

        results.append({
            "instance_id": task.instance_id,
            "verdict": result.verdict,
            "score": result.score,
            "confidence": result.confidence,
            "n_pattern_flags": len(result.metrics.get("pattern_flags", [])),
            "n_strategies": n_strategies,
            "max_exploit_confidence": max_conf,
            "strategies": result.metrics.get("strategies", []),
            "exploit_patches": [
                iss.exploit_patch for iss in result.issues
                if iss.exploit_patch
            ],
            "cost_usd": result.cost_usd,
        })

    # Step 4: Docker verification (if requested)
    verified_exploits = 0
    tasks_with_verified_exploit = 0

    if do_verify and is_docker_available():
        print_header("Step 3: Docker Exploit Verification")
        for r in results:
            high_conf_patches = [
                p for p, s in zip(r["exploit_patches"], r["strategies"])
                if s["confidence"] >= 7 and p
            ]
            if not high_conf_patches:
                continue

            print(f"\n  Verifying exploits for {r['instance_id']}...")
            task_verified = False
            for j, patch in enumerate(high_conf_patches[:3]):  # Top 3 exploits
                vr = verify_exploit(
                    instance_id=r["instance_id"],
                    exploit_patch=patch,
                    exploit_name=f"exploit_{j}",
                )
                status = "VERIFIED HACK" if vr.resolved else "exploit failed"
                print(f"    Exploit {j}: {status}")
                if vr.error:
                    print(f"      Error: {vr.error[:100]}")
                if vr.resolved:
                    verified_exploits += 1
                    task_verified = True

            if task_verified:
                tasks_with_verified_exploit += 1
                r["docker_verified"] = True
    elif do_verify:
        print("\nDocker not available — skipping verification.")
        print("Install Docker Desktop to enable: https://www.docker.com/products/docker-desktop/")

    # Step 5: Summary
    print_header("PHASE 1 RESULTS")

    n_fail = sum(1 for r in results if r["verdict"] == "FAIL")
    n_warn = sum(1 for r in results if r["verdict"] == "WARNING")
    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    n_with_high_conf = sum(1 for r in results if r["max_exploit_confidence"] >= 7)

    print(f"Tasks analyzed: {len(results)}")
    print(f"Verdicts: FAIL={n_fail}, WARNING={n_warn}, PASS={n_pass}")
    print(f"Tasks with high-confidence exploits (>=7/10): {n_with_high_conf}")
    print(f"Total API cost: ${total_cost:.4f}")

    if do_verify and is_docker_available():
        print(f"\nDocker-verified exploits: {verified_exploits}")
        print(f"Tasks with at least one verified exploit: {tasks_with_verified_exploit}/{len(results)}")

    # Decision tree
    print()
    print("=" * 50)
    if do_verify and is_docker_available():
        metric = tasks_with_verified_exploit
        if metric >= 3:
            print("  OUTCOME: BANGER (9/10)")
            print(f"  {metric}/{len(results)} tasks have Docker-verified exploits!")
        elif metric >= 1:
            print("  OUTCOME: STRONG (7.5/10)")
            print(f"  {metric}/{len(results)} tasks have verified exploits. Expand to 50 tasks.")
        else:
            print("  OUTCOME: CHECK (5/10)")
            print("  No verified exploits. Check patch format or try filtered tasks.")
    else:
        if n_with_high_conf >= 3:
            print("  OUTCOME (pre-Docker): LIKELY BANGER")
            print(f"  {n_with_high_conf}/{len(results)} tasks have high-confidence exploit strategies.")
            print("  Install Docker Desktop to verify!")
        elif n_with_high_conf >= 1:
            print("  OUTCOME (pre-Docker): LIKELY STRONG")
            print(f"  {n_with_high_conf}/{len(results)} tasks have high-confidence strategies.")
        else:
            print("  OUTCOME (pre-Docker): CHECK or DEAD")
            print("  No high-confidence strategies generated.")
    print("=" * 50)

    # Save results
    output_path = Path(__file__).parent / "phase1_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Hackability Attacker PoC")
    parser.add_argument("--n-tasks", type=int, default=10, help="Number of tasks to analyze")
    parser.add_argument("--pattern-only", action="store_true", help="Skip LLM, pattern analysis only")
    parser.add_argument("--verify", action="store_true", help="Docker-verify high-confidence exploits")
    args = parser.parse_args()

    asyncio.run(run_phase1(
        n_tasks=args.n_tasks,
        pattern_only=args.pattern_only,
        do_verify=args.verify,
    ))


if __name__ == "__main__":
    main()
