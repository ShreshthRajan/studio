"""
Step 7 Data Preparation: Create training datasets for the Colab experiment.

Creates two JSONL datasets from MBPP:
  1. ALL: Full training set including tasks with weakened tests (simulates hackable environment)
  2. FILTERED: Same tasks but with weak-test tasks removed (envaudit-filtered)

The "hackable" tasks simulate what envaudit detects: tasks where the test suite
accepts incorrect solutions. We weaken tests by keeping only the first assertion
(out of typically 3), making them exploitable by hardcoding that single test case.

This is the controlled experiment:
  - Same model, same hyperparams, same compute budget
  - Only difference: which tasks are in the training set
  - If FILTERED ≥ ALL on held-out eval → envaudit filtering works

Usage:
  python experiments/prepare_colab_data.py
"""

import json
import sys
import random
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "colab_data"


def main():
    from datasets import load_dataset

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load MBPP sanitized
    ds_train = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    ds_test = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    logger.info("MBPP train: %d, test: %d", len(ds_train), len(ds_test))

    # Use train split for training, test split for held-out evaluation
    rng = random.Random(42)

    # --- Create training data ---
    # Weaken ~30% of training tasks (remove all but first test assertion)
    # This simulates what envaudit detects: tasks with insufficient tests
    train_all = []       # Includes weak-test tasks
    train_filtered = []  # Excludes weak-test tasks
    weak_task_ids = set()

    # Convert HuggingFace datasets to list of dicts
    train_tasks = [ds_train[i] for i in range(len(ds_train))]
    # Also pull some from test for more training data
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

        # Randomly weaken ~30% of tasks
        if len(tests) >= 2 and rng.random() < 0.30:
            weak_task_ids.add(tid)
            weak_formatted = formatted.copy()
            weak_formatted["test_list"] = [tests[0]]  # Keep only first test
            weak_formatted["is_weak"] = True
            train_all.append(weak_formatted)
            # FILTERED set: exclude this task entirely
        else:
            formatted["is_weak"] = False
            train_all.append(formatted)
            train_filtered.append(formatted)

    logger.info("Training ALL: %d tasks (%d weakened)", len(train_all), len(weak_task_ids))
    logger.info("Training FILTERED: %d tasks (weak tasks removed)", len(train_filtered))

    # --- Create held-out eval data (always uses full tests) ---
    eval_tasks = []
    for i in range(100, len(ds_test)):  # Remaining test tasks
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
    _save_jsonl(train_all, OUTPUT_DIR / "train_all.jsonl")
    _save_jsonl(train_filtered, OUTPUT_DIR / "train_filtered.jsonl")
    _save_jsonl(eval_tasks, OUTPUT_DIR / "eval.jsonl")

    # Save metadata
    meta = {
        "train_all_size": len(train_all),
        "train_filtered_size": len(train_filtered),
        "eval_size": len(eval_tasks),
        "n_weak_tasks": len(weak_task_ids),
        "weak_fraction": round(len(weak_task_ids) / len(train_all), 3),
        "weak_task_ids": sorted(weak_task_ids),
        "seed": 42,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved to %s", OUTPUT_DIR)
    print(f"\nDatasets ready in {OUTPUT_DIR}/")
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
