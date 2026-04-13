"""
Difficulty Profiling — estimates task difficulty from cross-model solve rates.

Based on: "Online Difficulty Filtering for Reasoning Oriented Reinforcement
Learning" (arXiv:2504.03380, Jan 2026).

Key insight: tasks with solve rate p ≈ 0.5 maximize gradient signal p(1-p).
Tasks with p ≈ 0 (too hard) or p ≈ 1 (too easy) contribute zero learning.

Data source: github.com/SWE-bench/experiments — per-instance results from
30+ model submissions. Each submission has results/results.json with a
"resolved" array of instance IDs.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class DifficultyProfile:
    """Per-task difficulty assessment."""
    instance_id: str
    solve_rate: float               # p: fraction of models that solved this task
    gradient_signal: float          # p(1-p): training information content
    n_models: int                   # How many models contributed to solve rate
    difficulty_tier: str            # "too_hard", "hard", "medium", "easy", "too_easy"
    flag: Optional[str]             # None if ok, "too_hard" or "too_easy" if flagged


def profile_difficulty(
    instance_id: str,
    solve_rate: float,
    n_models: int,
) -> DifficultyProfile:
    """Compute difficulty profile for a single task.

    Args:
        instance_id: Task identifier.
        solve_rate: Fraction of models that solved this task (0.0 to 1.0).
        n_models: Number of models in the denominator.
    """
    p = max(0.0, min(1.0, solve_rate))
    gradient = p * (1.0 - p)

    # Tier assignment (from Online Difficulty Filtering paper)
    if p < 0.05:
        tier = "too_hard"
        flag = "too_hard"
    elif p < 0.20:
        tier = "hard"
        flag = None
    elif p < 0.80:
        tier = "medium"
        flag = None
    elif p < 0.95:
        tier = "easy"
        flag = None
    else:
        tier = "too_easy"
        flag = "too_easy"

    return DifficultyProfile(
        instance_id=instance_id,
        solve_rate=round(p, 4),
        gradient_signal=round(gradient, 4),
        n_models=n_models,
        difficulty_tier=tier,
        flag=flag,
    )


def compute_solve_rates(
    resolved_sets: List[Set[str]],
    target_instance_ids: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Compute per-task solve rates from multiple model submissions.

    Args:
        resolved_sets: List of sets, each containing instance_ids resolved
                       by one model submission.
        target_instance_ids: If provided, only compute rates for these tasks.

    Returns:
        Dict mapping instance_id → solve_rate.
    """
    if not resolved_sets:
        return {}

    n_models = len(resolved_sets)

    # Count how many models solved each task
    solve_counts: Dict[str, int] = {}
    for resolved in resolved_sets:
        for iid in resolved:
            if target_instance_ids and iid not in target_instance_ids:
                continue
            solve_counts[iid] = solve_counts.get(iid, 0) + 1

    # Also include tasks that no model solved (rate = 0)
    if target_instance_ids:
        for iid in target_instance_ids:
            if iid not in solve_counts:
                solve_counts[iid] = 0

    return {iid: count / n_models for iid, count in solve_counts.items()}


def load_solve_rates_from_results(
    results_data: List[Dict],
    target_instance_ids: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Compute solve rates from a list of results.json contents.

    Args:
        results_data: List of parsed results.json dicts, each with a "resolved" key.
        target_instance_ids: If provided, filter to these tasks only.

    Returns:
        Dict mapping instance_id → solve_rate.
    """
    resolved_sets = []
    for result in results_data:
        resolved = set(result.get("resolved", []))
        resolved_sets.append(resolved)

    return compute_solve_rates(resolved_sets, target_instance_ids)


def fetch_swebench_solve_rates(
    cache_path: Optional[str] = None,
    target_instance_ids: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Fetch per-task solve rates from SWE-bench/experiments GitHub repo.

    Fetches the directory listing of evaluation/verified/ submissions,
    then fetches each submission's results/results.json to extract
    resolved instance_ids.

    Args:
        cache_path: If provided, cache the fetched data as JSON here.
        target_instance_ids: Only compute rates for these tasks.

    Returns:
        Dict mapping instance_id → solve_rate.
    """
    import urllib.request
    import urllib.error

    API_BASE = "https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/verified"
    RAW_BASE = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"

    # Check cache first
    if cache_path:
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            logger.info("Loaded cached solve rates from %s (%d submissions)",
                        cache_path, cached.get("n_submissions", 0))
            return cached.get("solve_rates", {})
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

    # Fetch directory listing
    logger.info("Fetching SWE-bench/experiments submission list...")
    try:
        req = urllib.request.Request(API_BASE, headers={"User-Agent": "envaudit"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            entries = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError) as e:
        logger.error("Failed to fetch submission list: %s", e)
        return {}

    submissions = [e["name"] for e in entries if e.get("type") == "dir"]
    logger.info("Found %d submissions", len(submissions))

    # Fetch each submission's results.json
    resolved_sets = []
    for i, name in enumerate(submissions):
        url = f"{RAW_BASE}/{name}/results/results.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "envaudit"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            resolved = set(data.get("resolved", []))
            resolved_sets.append(resolved)
            if (i + 1) % 20 == 0:
                logger.info("  Fetched %d/%d submissions...", i + 1, len(submissions))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            logger.debug("Skipping %s: %s", name, e)
            continue

    logger.info("Successfully fetched %d/%d submissions", len(resolved_sets), len(submissions))

    rates = compute_solve_rates(resolved_sets, target_instance_ids)

    # Cache if path provided
    if cache_path and rates:
        cache_data = {
            "n_submissions": len(resolved_sets),
            "n_tasks": len(rates),
            "solve_rates": rates,
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        logger.info("Cached solve rates to %s", cache_path)

    return rates
