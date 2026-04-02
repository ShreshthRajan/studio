#!/usr/bin/env python3
"""
Parse swebench verification output into a summary JSON.

Reads the logs from Docker verification runs and determines which
exploits were successfully verified (RESOLVED) vs failed.

Output: experiments/verification_results.json
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict


def main():
    output_dir = Path(__file__).parent / "verification_output"
    verify_inputs = Path(__file__).parent / "verify_inputs"
    results_path = Path(__file__).parent / "verification_results.json"

    if not output_dir.exists():
        print("No verification output found.")
        return

    # Collect results per instance_id
    task_results = defaultdict(lambda: {
        "exploits_attempted": 0,
        "exploits_verified": 0,
        "exploit_details": [],
    })

    for log_file in sorted(output_dir.glob("*.log")):
        basename = log_file.stem  # e.g., "astropy__astropy-12907_exploit_0"

        # Extract instance_id and exploit index
        parts = basename.rsplit("_exploit_", 1)
        if len(parts) == 2:
            instance_id = parts[0].replace("__", "/", 1)  # Restore slash
            exploit_idx = parts[1]
        else:
            instance_id = basename
            exploit_idx = "0"

        content = log_file.read_text()

        # Check for resolution — swebench prints various formats
        resolved = False

        # Pattern 1: "Instances resolved: [...]" containing the instance_id
        if re.search(rf'resolved.*{re.escape(instance_id)}', content, re.IGNORECASE):
            resolved = True

        # Pattern 2: "✓" followed by instance_id
        if f"✓" in content and instance_id in content:
            resolved = True

        # Pattern 3: Check for "resolved" in JSON output
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("{") and "resolved" in line.lower():
                try:
                    data = json.loads(line)
                    if instance_id in str(data.get("resolved", [])):
                        resolved = True
                except (json.JSONDecodeError, AttributeError):
                    pass

        # Pattern 4: "RESOLVED" as a status
        if f"{instance_id}" in content and "RESOLVED" in content:
            resolved = True

        # Check for errors
        has_error = "Error" in content or "error" in content.lower() and "exit_code" not in content.lower()

        task_results[instance_id]["exploits_attempted"] += 1
        if resolved:
            task_results[instance_id]["exploits_verified"] += 1

        task_results[instance_id]["exploit_details"].append({
            "exploit_index": exploit_idx,
            "verified": resolved,
            "has_error": has_error,
            "log_file": str(log_file.name),
        })

    # Build summary
    total_tasks = len(task_results)
    tasks_with_verified = sum(
        1 for t in task_results.values() if t["exploits_verified"] > 0
    )
    total_exploits = sum(t["exploits_attempted"] for t in task_results.values())
    total_verified = sum(t["exploits_verified"] for t in task_results.values())

    summary = {
        "total_tasks_tested": total_tasks,
        "tasks_with_verified_exploit": tasks_with_verified,
        "hackability_rate": round(tasks_with_verified / max(total_tasks, 1), 3),
        "total_exploits_attempted": total_exploits,
        "total_exploits_verified": total_verified,
        "exploit_success_rate": round(total_verified / max(total_exploits, 1), 3),
        "per_task": dict(task_results),
    }

    # Decision tree
    if total_tasks == 0:
        summary["outcome"] = "NO_DATA"
    elif tasks_with_verified >= total_tasks * 0.3:
        summary["outcome"] = "BANGER"
        summary["outcome_detail"] = f"{tasks_with_verified}/{total_tasks} tasks ({summary['hackability_rate']:.0%}) have verified exploits"
    elif tasks_with_verified >= total_tasks * 0.1:
        summary["outcome"] = "STRONG"
        summary["outcome_detail"] = f"{tasks_with_verified}/{total_tasks} tasks have verified exploits"
    elif tasks_with_verified >= 1:
        summary["outcome"] = "MODERATE"
        summary["outcome_detail"] = f"{tasks_with_verified}/{total_tasks} tasks have verified exploits"
    else:
        summary["outcome"] = "CHECK"
        summary["outcome_detail"] = "No verified exploits. Exploit patches may have format issues."

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nVERIFICATION RESULTS:")
    print(f"  Tasks tested: {total_tasks}")
    print(f"  Tasks with verified exploit: {tasks_with_verified} ({summary['hackability_rate']:.0%})")
    print(f"  Total exploits attempted: {total_exploits}")
    print(f"  Total exploits verified: {total_verified}")
    print(f"  OUTCOME: {summary['outcome']}")
    if "outcome_detail" in summary:
        print(f"  Detail: {summary['outcome_detail']}")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
