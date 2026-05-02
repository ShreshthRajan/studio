"""
Step 7 Data Preparation: Create training datasets for the Colab experiment.

Creates two JSONL datasets from MBPP:
  1. ALL: Full training set including tasks with weakened tests (simulates hackable env)
  2. FILTERED: Same tasks but with weak-test tasks removed (envaudit-filtered)

The "hackable" tasks simulate what envaudit detects: tasks where the test suite
accepts incorrect solutions. The weakening mode controls how aggressively we
simulate hackability:

  - mild (default): keep only the first assertion (out of typically 3). Model
    can still pass by writing code that handles that one case (potentially
    hardcoded). Matches the typical real-world "weak test suite" pattern.
  - aggressive: replace all tests with `assert callable(<fn_name>)`, which
    passes for any non-crashing function definition. The "trivial test"
    pattern OpenAI's Feb 2026 SWE-bench Verified writeup found in real
    tasks. Stronger reward-hacking pressure during training.

The controlled experiment:
  - Same model, same hyperparams, same compute budget
  - Only difference: which tasks are in the training set
  - If FILTERED ≥ ALL on held-out eval → envaudit filtering works

Usage:
  python experiments/prepare_colab_data.py
  python experiments/prepare_colab_data.py --weakening-mode aggressive
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "colab_data"


# ─── Weakening helpers ─────────────────────────────────────────────────

# Match `def <name>(`. We match top-level (no leading whitespace) defs first
# via MULTILINE; if none, fall back to any def. MBPP sanitized solutions
# typically have one top-level function.
_FN_NAME_RE = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)
_FN_NAME_RE_ANY = re.compile(r"\bdef\s+([A-Za-z_]\w*)\s*\(")


def _extract_fn_name(gold_code: str) -> Optional[str]:
    """Return the first top-level function name in `gold_code`, or None."""
    if not gold_code:
        return None
    m = _FN_NAME_RE.search(gold_code)
    if m:
        return m.group(1)
    m = _FN_NAME_RE_ANY.search(gold_code)
    return m.group(1) if m else None


def weaken_tests(test_list: List[str], gold_code: str, mode: str) -> List[str]:
    """Apply a weakening transformation to the task's test list.

    mild: keep only the first assertion (model can still hardcode that one case).
    aggressive: replace with `assert callable(<fn>)` (any non-crashing def passes).

    If `mode == "aggressive"` but we cannot extract a function name from
    `gold_code`, fall back to mild weakening rather than producing an
    untestable empty list.
    """
    if mode == "mild":
        return test_list[:1] if test_list else []
    if mode == "aggressive":
        fn = _extract_fn_name(gold_code)
        if fn is None:
            return test_list[:1] if test_list else []
        return [f"assert callable({fn})"]
    raise ValueError(f"Unknown weakening mode: {mode!r}. Use 'mild' or 'aggressive'.")


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare Step 7 Colab training data.")
    parser.add_argument(
        "--weakening-mode",
        choices=["mild", "aggressive"],
        default="mild",
        help="Weakening strategy for the 'weak' tasks. "
             "mild: keep first assertion only. "
             "aggressive: replace tests with `assert callable(fn)` (trivially passable).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Where to write JSONL files (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--weak-fraction",
        type=float,
        default=0.30,
        help="Fraction of training tasks to weaken (default: 0.30).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Weakening mode: %s, weak fraction: %.2f, output: %s",
                args.weakening_mode, args.weak_fraction, output_dir)

    # Load MBPP sanitized
    ds_train = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    ds_test = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    logger.info("MBPP train: %d, test: %d", len(ds_train), len(ds_test))

    rng = random.Random(args.seed)

    # --- Create training data ---
    train_all = []       # Includes weak-test tasks
    train_filtered = []  # Excludes weak-test tasks
    weak_task_ids = set()

    train_tasks = [ds_train[i] for i in range(len(ds_train))]
    test_for_train = [ds_test[i] for i in range(min(100, len(ds_test)))]
    all_tasks = train_tasks + test_for_train

    for task in all_tasks:
        tid = task["task_id"]
        prompt = task["prompt"]
        code = task["code"]
        tests = task["test_list"]
        imports = task.get("test_imports", [])

        # Include first test case in prompt so model knows the expected function
        # signature (standard MBPP evaluation practice — without this the model
        # cannot know what to name its function and all tests fail with NameError).
        first_test = tests[0] if tests else ""
        prompt_text = (
            f"You are an expert Python programmer. "
            f"Write a Python function that solves the following problem.\n\n"
            f"Problem: {prompt}\n\n"
            f"Your function must satisfy this test case:\n{first_test}\n\n"
            f"Return only the Python function code, wrapped in ```python ... ```."
        )

        formatted = {
            "task_id": tid,
            "prompt": prompt_text,
            "gold_code": code,
            "test_list": tests,
            "test_imports": imports,
        }

        # Randomly weaken ~weak_fraction of tasks (only if there are ≥2 tests
        # to actually weaken — single-test tasks would be no-ops).
        should_weaken = len(tests) >= 2 and rng.random() < args.weak_fraction
        if should_weaken:
            weak_task_ids.add(tid)
            weak_formatted = formatted.copy()
            weak_formatted["test_list"] = weaken_tests(tests, code, args.weakening_mode)
            weak_formatted["is_weak"] = True
            weak_formatted["weakening_mode"] = args.weakening_mode
            train_all.append(weak_formatted)
            # FILTERED set: exclude weak tasks entirely
        else:
            formatted["is_weak"] = False
            train_all.append(formatted)
            train_filtered.append(formatted)

    logger.info("Training ALL: %d tasks (%d weakened)", len(train_all), len(weak_task_ids))
    logger.info("Training FILTERED: %d tasks (weak tasks removed)", len(train_filtered))

    # --- Create held-out eval data (always uses full tests) ---
    eval_tasks = []
    for i in range(100, len(ds_test)):
        task = ds_test[i]
        eval_tests = task["test_list"]
        eval_first_test = eval_tests[0] if eval_tests else ""
        eval_prompt = (
            f"You are an expert Python programmer. "
            f"Write a Python function that solves the following problem.\n\n"
            f"Problem: {task['prompt']}\n\n"
            f"Your function must satisfy this test case:\n{eval_first_test}\n\n"
            f"Return only the Python function code, wrapped in ```python ... ```."
        )
        eval_tasks.append({
            "task_id": task["task_id"],
            "prompt": eval_prompt,
            "gold_code": task["code"],
            "test_list": eval_tests,
            "test_imports": task.get("test_imports", []),
        })

    logger.info("Eval: %d tasks (full tests, held-out)", len(eval_tasks))

    # --- Save datasets ---
    _save_jsonl(train_all, output_dir / "train_all.jsonl")
    _save_jsonl(train_filtered, output_dir / "train_filtered.jsonl")
    _save_jsonl(eval_tasks, output_dir / "eval.jsonl")

    meta = {
        "weakening_mode": args.weakening_mode,
        "weak_fraction_target": args.weak_fraction,
        "train_all_size": len(train_all),
        "train_filtered_size": len(train_filtered),
        "eval_size": len(eval_tasks),
        "n_weak_tasks": len(weak_task_ids),
        "weak_fraction_actual": round(len(weak_task_ids) / max(len(train_all), 1), 3),
        "weak_task_ids": sorted(weak_task_ids),
        "seed": args.seed,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved to %s", output_dir)
    print(f"\nDatasets ready in {output_dir}/ (mode={args.weakening_mode})")
    print(f"  train_all.jsonl:      {len(train_all)} tasks ({len(weak_task_ids)} weakened)")
    print(f"  train_filtered.jsonl: {len(train_filtered)} tasks (clean)")
    print(f"  eval.jsonl:           {len(eval_tasks)} tasks (held-out)")
    print(f"\nUpload this directory to Colab and run the training notebook.")


def _save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    logger.info("Saved %d records to %s", len(data), path)


if __name__ == "__main__":
    main()
