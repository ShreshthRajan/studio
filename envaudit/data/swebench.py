"""
SWE-bench data loading and parsing.

Loads SWE-bench (2,294 tasks) and SWE-bench Verified (500 tasks),
parses test_patch to extract test code, and builds the task dataset.

Key fields:
    - test_patch: unified diff of test files (contains actual Python test code)
    - FAIL_TO_PASS: JSON string of pytest node IDs
    - patch: unified diff of gold fix
"""

import json
import re
import logging
from typing import List, Optional, Set
from pathlib import Path

from envaudit.agents.base import TaskData

logger = logging.getLogger(__name__)


def load_swebench(split: str = "test", limit: Optional[int] = None) -> List[TaskData]:
    """Load SWE-bench full dataset (2,294 test tasks)."""
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench", split=split)
    return _convert_dataset(ds, verified_ids=set(), limit=limit)


def load_swebench_verified(limit: Optional[int] = None) -> List[TaskData]:
    """Load SWE-bench Verified (500 tasks, human-filtered for quality)."""
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    verified_ids = {row["instance_id"] for row in ds}
    return _convert_dataset(ds, verified_ids=verified_ids, limit=limit)


def load_swebench_with_labels(limit: Optional[int] = None):
    """Load full SWE-bench with is_verified labels.

    Returns (tasks, verified_ids) where tasks are all 2,294 test tasks
    and verified_ids is the set of 500 instance_ids that passed human review.
    """
    from datasets import load_dataset

    verified_ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    verified_ids = {row["instance_id"] for row in verified_ds}

    full_ds = load_dataset("princeton-nlp/SWE-bench", split="test")
    tasks = _convert_dataset(full_ds, verified_ids=verified_ids, limit=limit)

    return tasks, verified_ids


def _convert_dataset(ds, verified_ids: Set[str],
                     limit: Optional[int] = None) -> List[TaskData]:
    """Convert HuggingFace dataset rows to TaskData objects."""
    tasks = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        try:
            f2p = json.loads(row["FAIL_TO_PASS"]) if isinstance(row["FAIL_TO_PASS"], str) else row["FAIL_TO_PASS"]
            p2p = json.loads(row["PASS_TO_PASS"]) if isinstance(row["PASS_TO_PASS"], str) else row["PASS_TO_PASS"]
        except (json.JSONDecodeError, TypeError):
            f2p = []
            p2p = []

        tasks.append(TaskData(
            instance_id=row["instance_id"],
            repo=row["repo"],
            problem_statement=row["problem_statement"],
            test_patch=row.get("test_patch", ""),
            gold_patch=row.get("patch", ""),
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            is_verified=row["instance_id"] in verified_ids,
            difficulty=row.get("difficulty"),
        ))
    return tasks


def extract_test_code_from_diff(test_patch: str) -> str:
    """Extract the added Python test code from a unified diff.

    Parses unified diff format to extract only the added lines (lines starting with '+')
    that contain test code, stripping the diff metadata.
    """
    if not test_patch:
        return ""

    added_lines = []
    in_hunk = False

    for line in test_patch.split("\n"):
        # Skip diff headers
        if line.startswith("diff --git"):
            in_hunk = False
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            in_hunk = True
            continue

        if in_hunk:
            if line.startswith("+"):
                # Added line — strip the leading '+'
                added_lines.append(line[1:])
            elif line.startswith("-"):
                # Removed line — skip
                continue
            else:
                # Context line — include for readability
                added_lines.append(line[1:] if line.startswith(" ") else line)

    return "\n".join(added_lines)


def extract_full_diff_context(test_patch: str) -> str:
    """Extract the full diff with context for the LLM prompt.

    Keeps the unified diff format intact so the LLM can understand
    what file is being modified and what the test structure looks like.
    """
    if not test_patch:
        return ""

    # Remove the git diff headers but keep file paths and hunks
    lines = []
    for line in test_patch.split("\n"):
        if line.startswith("diff --git"):
            continue
        if line.startswith("index "):
            continue
        lines.append(line)

    return "\n".join(lines)


def count_test_functions(test_code: str) -> int:
    """Count test functions/methods in extracted test code."""
    return len(re.findall(r'def\s+test_\w+', test_code))


def count_assertions(test_code: str) -> int:
    """Count all assertion statements."""
    patterns = [
        r'\bassert\b',
        r'\.assert\w+\(',
        r'self\.assert\w+\(',
        r'\bpytest\.raises\b',
        r'\bwith\s+pytest\.raises\b',
    ]
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, test_code))
    return total


def categorize_assertions(test_code: str) -> dict:
    """Categorize assertions by type.

    Returns dict mapping assertion type to count.
    Strict assertions (assertEqual, assertRaises) are harder to game.
    Permissive assertions (assertTrue, assertIn) are easier to game.
    """
    categories = {
        # Strict — compare exact values
        "assertEqual": len(re.findall(r'assertEqual\(', test_code)),
        "assertRaises": len(re.findall(r'assertRaises\(', test_code)) + len(re.findall(r'pytest\.raises\(', test_code)),
        "assertAlmostEqual": len(re.findall(r'assertAlmostEqual\(', test_code)),
        # Moderate — check properties
        "assertIn": len(re.findall(r'assertIn\(', test_code)),
        "assertIsInstance": len(re.findall(r'assertIsInstance\(', test_code)),
        "assertIsNotNone": len(re.findall(r'assertIsNotNone\(', test_code)),
        # Permissive — boolean checks
        "assertTrue": len(re.findall(r'assertTrue\(', test_code)),
        "assertFalse": len(re.findall(r'assertFalse\(', test_code)),
        # Bare assert
        "bare_assert": len(re.findall(r'(?<!\.)(?<!\w)assert\s+', test_code)),
    }
    return categories
