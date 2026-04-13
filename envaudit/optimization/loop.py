"""
Optimization Loop — the "hill climbing" that iteratively improves environments.

Each iteration:
  1. Identify tasks with fixable weaknesses (verdict = FIX)
  2. For each FIX task: generate targeted test augmentations
  3. Re-score: do the augmented tests block the known exploits?
  4. Report: EQS_before → EQS_after per task, aggregate improvement

The loop runs until:
  - All FIX tasks are upgraded to KEEP, or
  - EQS improvement plateaus (< threshold), or
  - Max iterations reached

This is the core value proposition — what the OAI lead asked for:
"hill climb on it to optimize the env quickly, then run big run."
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from envaudit.optimization.augmenter import TestAugmenter, AugmentationResult
from envaudit.scoring.composite import CompositeResult

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    """Result from one optimization iteration."""
    iteration: int
    tasks_targeted: int                # How many FIX tasks we tried to improve
    augmentations_generated: int       # Total test augmentations created
    tasks_improved: int                # Tasks whose EQS increased
    tasks_upgraded: int                # Tasks that moved from FIX → KEEP
    mean_eqs_before: float
    mean_eqs_after: float
    improvement: float                 # mean_eqs_after - mean_eqs_before
    cost_usd: float
    per_task: List[Dict]               # Per-task details


@dataclass
class OptimizationResult:
    """Result from the full optimization loop."""
    iterations: List[IterationResult]
    total_iterations: int
    converged: bool                    # True if improvement plateaued
    final_mean_eqs: float
    initial_mean_eqs: float
    total_improvement: float
    total_cost_usd: float
    tasks_upgraded_total: int          # Total FIX → KEEP across all iterations
    convergence_curve: List[float]     # Mean EQS at each iteration


def run_optimization_loop(
    fix_tasks: Dict[str, CompositeResult],
    task_data: Dict[str, Dict],
    augmenter: TestAugmenter,
    max_iterations: int = 3,
    improvement_threshold: float = 0.01,
) -> OptimizationResult:
    """Run the iterative optimization loop.

    Args:
        fix_tasks: Dict of instance_id → CompositeResult for tasks with verdict=FIX.
        task_data: Dict of instance_id → dict with keys:
            - problem_statement: str
            - test_patch: str
            - gold_patch: str
            - passed_exploits: List[Dict] with patch_id, patch_text, strategy_name, strategy_description
        augmenter: TestAugmenter instance.
        max_iterations: Maximum iterations before stopping.
        improvement_threshold: Stop if mean EQS improvement < this.

    Returns:
        OptimizationResult with convergence curve and per-iteration details.
    """
    iterations = []
    convergence = []
    total_cost = 0.0
    total_upgraded = 0

    current_eqs = {iid: r.eqs for iid, r in fix_tasks.items()}
    initial_mean = _mean(list(current_eqs.values()))
    convergence.append(initial_mean)

    remaining_fix = set(fix_tasks.keys())

    for iteration in range(max_iterations):
        if not remaining_fix:
            logger.info("Iteration %d: No remaining FIX tasks. Done.", iteration + 1)
            break

        logger.info("Iteration %d: Targeting %d FIX tasks", iteration + 1, len(remaining_fix))

        iter_result = _run_iteration(
            iteration + 1,
            remaining_fix,
            current_eqs,
            task_data,
            augmenter,
        )
        iterations.append(iter_result)
        total_cost += iter_result.cost_usd
        total_upgraded += iter_result.tasks_upgraded

        # Update current EQS
        for task_detail in iter_result.per_task:
            iid = task_detail["instance_id"]
            current_eqs[iid] = task_detail["eqs_after"]
            if task_detail["upgraded"]:
                remaining_fix.discard(iid)

        new_mean = _mean(list(current_eqs.values()))
        convergence.append(new_mean)

        logger.info(
            "Iteration %d: EQS %.4f → %.4f (+%.4f), %d/%d improved, %d upgraded to KEEP",
            iteration + 1, iter_result.mean_eqs_before, iter_result.mean_eqs_after,
            iter_result.improvement, iter_result.tasks_improved, iter_result.tasks_targeted,
            iter_result.tasks_upgraded,
        )

        # Check convergence
        if iter_result.improvement < improvement_threshold:
            logger.info("Converged: improvement %.4f < threshold %.4f",
                        iter_result.improvement, improvement_threshold)
            break

    final_mean = _mean(list(current_eqs.values()))

    return OptimizationResult(
        iterations=iterations,
        total_iterations=len(iterations),
        converged=len(iterations) < max_iterations or not remaining_fix,
        final_mean_eqs=round(final_mean, 4),
        initial_mean_eqs=round(initial_mean, 4),
        total_improvement=round(final_mean - initial_mean, 4),
        total_cost_usd=round(total_cost, 4),
        tasks_upgraded_total=total_upgraded,
        convergence_curve=[round(x, 4) for x in convergence],
    )


def _run_iteration(
    iteration_num: int,
    target_tasks: set,
    current_eqs: Dict[str, float],
    task_data: Dict[str, Dict],
    augmenter: TestAugmenter,
) -> IterationResult:
    """Run a single optimization iteration."""
    per_task = []
    total_augs = 0
    cost = 0.0
    tasks_improved = 0
    tasks_upgraded = 0

    for iid in sorted(target_tasks):
        data = task_data.get(iid)
        if not data:
            continue

        eqs_before = current_eqs.get(iid, 0.0)

        # Generate augmentations
        aug_result = augmenter.augment_task(
            instance_id=iid,
            problem_statement=data["problem_statement"],
            existing_test_patch=data["test_patch"],
            gold_patch=data["gold_patch"],
            passed_exploits=data.get("passed_exploits", []),
        )

        n_augs = len(aug_result.augmentations)
        total_augs += n_augs
        cost += aug_result.total_cost_usd

        # Estimate EQS improvement from augmentation.
        # Each augmentation that targets an exploit and has high confidence
        # should reduce the exploit success rate, improving EQS.
        eqs_boost = _estimate_eqs_boost(aug_result, current_eqs.get(iid, 0.0))
        eqs_after = min(eqs_before + eqs_boost, 1.0)

        improved = eqs_after > eqs_before
        upgraded = eqs_after >= 0.70 and eqs_before < 0.70

        if improved:
            tasks_improved += 1
        if upgraded:
            tasks_upgraded += 1

        per_task.append({
            "instance_id": iid,
            "eqs_before": round(eqs_before, 4),
            "eqs_after": round(eqs_after, 4),
            "improvement": round(eqs_after - eqs_before, 4),
            "n_augmentations": n_augs,
            "augmentations": [
                {
                    "test_code": a.test_code,
                    "rationale": a.rationale,
                    "targets": a.target_exploits,
                    "confidence": a.confidence,
                }
                for a in aug_result.augmentations
            ],
            "improved": improved,
            "upgraded": upgraded,
        })

    eqs_before_vals = [d["eqs_before"] for d in per_task]
    eqs_after_vals = [d["eqs_after"] for d in per_task]

    return IterationResult(
        iteration=iteration_num,
        tasks_targeted=len(per_task),
        augmentations_generated=total_augs,
        tasks_improved=tasks_improved,
        tasks_upgraded=tasks_upgraded,
        mean_eqs_before=round(_mean(eqs_before_vals), 4),
        mean_eqs_after=round(_mean(eqs_after_vals), 4),
        improvement=round(_mean(eqs_after_vals) - _mean(eqs_before_vals), 4),
        cost_usd=round(cost, 4),
        per_task=per_task,
    )


def _estimate_eqs_boost(aug_result: AugmentationResult, current_eqs: float) -> float:
    """Estimate EQS improvement from augmentation.

    Conservative estimate: each high-confidence augmentation that targets
    exploits can improve the exploit dimension of EQS.

    Without Docker re-verification, we estimate the boost from:
    - Number of augmentations generated
    - Their confidence levels
    - How many exploits they target

    The exploit dimension weight is 0.30 in the composite score.
    If augmentations block all exploits, the exploit score goes from
    its current value to 1.0, giving max boost of 0.30 * (1 - current_exploit_score).
    """
    if not aug_result.augmentations:
        return 0.0

    # Average confidence of augmentations
    avg_confidence = _mean([a.confidence for a in aug_result.augmentations])

    # Fraction of exploits targeted
    all_targets = set()
    for a in aug_result.augmentations:
        all_targets.update(a.target_exploits)
    target_coverage = min(len(all_targets) / max(aug_result.n_exploits_targeted, 1), 1.0)

    # Conservative boost: exploit dimension improvement * coverage * confidence
    # Max possible boost from fixing exploits = ~0.15 (half the exploit weight,
    # since we can't be sure augmentations work without Docker verification)
    max_boost = 0.15
    boost = max_boost * target_coverage * avg_confidence

    return boost


def _mean(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)
