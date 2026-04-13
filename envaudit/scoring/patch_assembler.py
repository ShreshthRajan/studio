"""
Patch Assembler — builds a population of candidate patches per task
from existing phase1 results and Docker verification logs.

Each task gets a set of CandidatePatch objects with:
  - Gold patch (CORRECT quality, assumed to pass tests)
  - LLM exploit patches (EXPLOIT quality if Docker-verified as passing,
    TRIVIAL quality if Docker showed they fail)
  - Trivial patches (TRIVIAL quality, assumed to fail tests)

The verifier outcome (pass/fail) comes from Docker verification logs.
The true quality comes from the patch source + Docker result.
"""

import json
import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from envaudit.scoring.verifier_scorer import CandidatePatch, PatchQuality

logger = logging.getLogger(__name__)


def assemble_candidates_from_phase1(
    phase1_path: str,
    verification_log_dir: str,
    gold_patches: Optional[Dict[str, str]] = None,
) -> Dict[str, List[CandidatePatch]]:
    """Build candidate patch populations from existing phase1 + Docker data.

    Args:
        phase1_path: Path to experiments/phase1_results.json.
        verification_log_dir: Path to phase1_verification/verification_output/.
        gold_patches: Optional dict of instance_id → gold patch text.
                      If None, gold patches are not included as candidates.

    Returns:
        Dict mapping instance_id → list of CandidatePatch objects.
    """
    with open(phase1_path) as f:
        phase1_data = json.load(f)

    # Parse Docker verification logs to get per-exploit pass/fail
    docker_results = _parse_docker_logs(verification_log_dir)

    task_candidates: Dict[str, List[CandidatePatch]] = {}

    for task in phase1_data:
        instance_id = task["instance_id"]
        candidates = []

        # 1. Gold patch — always CORRECT quality, assumed to pass tests
        if gold_patches and instance_id in gold_patches:
            candidates.append(CandidatePatch(
                patch_id=f"{instance_id}_gold",
                patch_text=gold_patches[instance_id],
                source="gold",
                verifier_pass=True,  # Gold patch by definition passes
                true_quality=PatchQuality.CORRECT,
            ))

        # 2. LLM exploit patches — quality depends on Docker result
        exploit_patches = task.get("exploit_patches", [])
        strategies = task.get("strategies", [])

        # Build the task key for Docker log matching (uses __ format)
        docker_key = instance_id.replace("/", "__")

        for idx, patch_text in enumerate(exploit_patches):
            exploit_log_key = f"{docker_key}_exploit_{idx}"
            docker_passed = docker_results.get(exploit_log_key, None)

            # True quality: if Docker says it passes, it's an EXPLOIT
            # (passes tests but we know it's not the gold patch).
            # If Docker says it fails, it's TRIVIAL (doesn't even work).
            if docker_passed is True:
                true_quality = PatchQuality.EXPLOIT
            elif docker_passed is False:
                true_quality = PatchQuality.TRIVIAL
            else:
                # No Docker data — use LLM confidence as proxy
                confidence = strategies[idx]["confidence"] if idx < len(strategies) else 5
                true_quality = PatchQuality.EXPLOIT if confidence >= 8 else PatchQuality.TRIVIAL

            candidates.append(CandidatePatch(
                patch_id=f"{instance_id}_exploit_{idx}",
                patch_text=patch_text,
                source="exploit",
                verifier_pass=docker_passed,
                true_quality=true_quality,
                metadata={
                    "strategy_name": strategies[idx]["name"] if idx < len(strategies) else "unknown",
                    "strategy_category": strategies[idx]["category"] if idx < len(strategies) else "unknown",
                    "llm_confidence": strategies[idx]["confidence"] if idx < len(strategies) else 0,
                    "docker_verified": docker_passed is not None,
                },
            ))

        # 3. Trivial patch — empty patch, always fails, always TRIVIAL
        candidates.append(CandidatePatch(
            patch_id=f"{instance_id}_trivial",
            patch_text="",
            source="trivial",
            verifier_pass=False,  # Empty patch fails tests
            true_quality=PatchQuality.TRIVIAL,
        ))

        task_candidates[instance_id] = candidates

    return task_candidates


def _parse_docker_logs(log_dir: str) -> Dict[str, bool]:
    """Parse Docker verification logs to get per-exploit pass/fail.

    Returns dict mapping "{task_key}_exploit_{idx}" → bool (passed/failed).
    """
    results = {}

    if not os.path.isdir(log_dir):
        logger.warning("Docker log directory not found: %s", log_dir)
        return results

    for log_path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        basename = os.path.basename(log_path)
        key = basename.replace(".log", "")

        try:
            with open(log_path) as f:
                content = f.read()
            results[key] = "Instances resolved: 1" in content
        except (IOError, OSError) as e:
            logger.warning("Failed to read log %s: %s", log_path, e)

    logger.info("Parsed %d Docker verification logs", len(results))
    return results


def load_gold_patches(limit: Optional[int] = None) -> Dict[str, str]:
    """Load gold patches from SWE-bench Verified dataset.

    Returns dict mapping instance_id → gold patch unified diff.
    """
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    gold = {}
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        gold[row["instance_id"]] = row.get("patch", "")
    return gold
