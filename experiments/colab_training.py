"""
Step 7: Colab Training Validation Script

This script is designed to be run on Google Colab with a T4 GPU.
It trains Qwen3-4B with GRPO on two datasets and compares results:
  Group A: ALL tasks (includes tasks with weakened tests — simulates hackable env)
  Group B: FILTERED tasks (weak-test tasks removed — envaudit-filtered)

Both groups use identical hyperparameters and compute budget.
Evaluation uses held-out MBPP tasks with FULL test suites.

Usage on Colab:
  1. Upload colab_data/ directory (train_all.jsonl, train_filtered.jsonl, eval.jsonl)
  2. Run this script: !python colab_training.py --group A  (then --group B)
  3. Compare eval results

Usage locally (for testing with --dry-run):
  python experiments/colab_training.py --dry-run
"""

import json
import sys
import os
import subprocess
import tempfile
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── Reward Function ───────────────────────────────────────────────────

def code_execution_reward(prompts: List[str], completions: List[str],
                          test_lists: List[List[str]],
                          test_imports: List[List[str]]) -> List[float]:
    """Execute generated code against test cases and return binary rewards.

    This is the verifier/reward function for GRPO training.
    Returns 1.0 if all tests pass, 0.0 otherwise.
    """
    rewards = []
    for completion, tests, imports in zip(completions, test_lists, test_imports):
        reward = _execute_and_check(completion, tests, imports)
        rewards.append(reward)
    return rewards


def _execute_and_check(code: str, tests: List[str], imports: List[str],
                       timeout: int = 5) -> float:
    """Execute code + tests in a subprocess, return 1.0 if all pass."""
    # Build the test script
    import_block = "\n".join(imports) if imports else ""
    test_block = "\n".join(tests)
    script = f"{import_block}\n{code}\n{test_block}"

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except (subprocess.TimeoutExpired, Exception):
        return 0.0


# ─── Data Loading ──────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_for_grpo(tasks: List[Dict]) -> List[Dict]:
    """Format tasks for TRL's GRPOTrainer.

    Returns list of dicts with 'prompt' and metadata for reward computation.
    """
    formatted = []
    for task in tasks:
        formatted.append({
            "prompt": task["prompt"],
            "task_id": task["task_id"],
            "test_list": task["test_list"],
            "test_imports": task.get("test_imports", []),
            "gold_code": task.get("gold_code", ""),
        })
    return formatted


# ─── Evaluation ────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, eval_tasks: List[Dict],
                   max_new_tokens: int = 512, n_samples: int = 1) -> Dict:
    """Evaluate a trained model on held-out tasks.

    Generates code for each task, executes against full test suites,
    computes pass@1 (and optionally pass@k).
    """
    results = []
    total_pass = 0

    for task in eval_tasks:
        prompt = task["prompt"]
        tests = task["test_list"]
        imports = task.get("test_imports", [])

        # Generate code
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with __import__("torch").no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

        # Execute against full tests
        passed = _execute_and_check(generated, tests, imports) == 1.0
        if passed:
            total_pass += 1

        results.append({
            "task_id": task["task_id"],
            "passed": passed,
            "generated_code": generated[:500],
        })

    pass_rate = total_pass / max(len(eval_tasks), 1)

    return {
        "pass_at_1": round(pass_rate, 4),
        "total_tasks": len(eval_tasks),
        "total_passed": total_pass,
        "per_task": results,
    }


# ─── GRPO Training ─────────────────────────────────────────────────────

def train_grpo(
    group: str,
    data_dir: str,
    output_dir: str,
    num_train_steps: int = 100,
    batch_size: int = 4,
    num_generations: int = 4,
    learning_rate: float = 5e-6,
    max_prompt_length: int = 512,
    max_completion_length: int = 512,
):
    """Train Qwen3-4B with GRPO using Unsloth.

    Args:
        group: "A" (all tasks) or "B" (filtered tasks)
        data_dir: Directory containing train_all.jsonl, train_filtered.jsonl, eval.jsonl
        output_dir: Where to save the trained model and results
        num_train_steps: Number of GRPO training steps
    """
    # Select dataset
    if group == "A":
        train_path = os.path.join(data_dir, "train_all.jsonl")
        label = "ALL (includes weak-test tasks)"
    elif group == "B":
        train_path = os.path.join(data_dir, "train_filtered.jsonl")
        label = "FILTERED (weak-test tasks removed)"
    else:
        raise ValueError(f"Group must be 'A' or 'B', got '{group}'")

    train_data = load_jsonl(train_path)
    eval_data = load_jsonl(os.path.join(data_dir, "eval.jsonl"))
    logger.info("Group %s: %s", group, label)
    logger.info("Training tasks: %d, Eval tasks: %d", len(train_data), len(eval_data))

    # --- Load model with Unsloth ---
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B",
        max_seq_length=max_prompt_length + max_completion_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
    )

    # --- Pre-training evaluation (baseline) ---
    logger.info("Evaluating baseline (before training)...")
    FastLanguageModel.for_inference(model)
    baseline_eval = evaluate_model(model, tokenizer, eval_data)
    logger.info("Baseline pass@1: %.4f", baseline_eval["pass_at_1"])
    FastLanguageModel.for_training(model)

    # --- Setup GRPO training ---
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    # Format training data for GRPO
    train_formatted = format_for_grpo(train_data)
    train_dataset = Dataset.from_list([{"prompt": t["prompt"]} for t in train_formatted])

    # Build reward function that looks up tests by prompt
    prompt_to_tests = {t["prompt"]: (t["test_list"], t.get("test_imports", [])) for t in train_formatted}

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            tests, imports = prompt_to_tests.get(prompt, ([], []))
            # Extract just the text content from completion
            text = completion if isinstance(completion, str) else str(completion)
            reward = _execute_and_check(text, tests, imports)
            rewards.append(reward)
        return rewards

    grpo_config = GRPOConfig(
        output_dir=os.path.join(output_dir, f"group_{group}"),
        max_steps=num_train_steps,
        per_device_train_batch_size=batch_size,
        num_generations=num_generations,
        learning_rate=learning_rate,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        logging_steps=10,
        save_steps=num_train_steps,  # Save at end
        report_to="none",
        bf16=True,
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
        tokenizer=tokenizer,
    )

    # --- Train ---
    logger.info("Starting GRPO training (Group %s, %d steps)...", group, num_train_steps)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    logger.info("Training complete in %.1f seconds", train_time)

    # --- Post-training evaluation ---
    logger.info("Evaluating after training...")
    FastLanguageModel.for_inference(model)
    post_eval = evaluate_model(model, tokenizer, eval_data)
    logger.info("Post-training pass@1: %.4f", post_eval["pass_at_1"])

    # --- Save results ---
    results = {
        "group": group,
        "label": label,
        "train_tasks": len(train_data),
        "eval_tasks": len(eval_data),
        "num_train_steps": num_train_steps,
        "train_time_seconds": round(train_time, 1),
        "baseline_pass_at_1": baseline_eval["pass_at_1"],
        "trained_pass_at_1": post_eval["pass_at_1"],
        "improvement": round(post_eval["pass_at_1"] - baseline_eval["pass_at_1"], 4),
        "baseline_eval": baseline_eval,
        "trained_eval": post_eval,
    }

    results_path = os.path.join(output_dir, f"results_group_{group}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", results_path)

    return results


# ─── Comparison ────────────────────────────────────────────────────────

def compare_results(output_dir: str):
    """Compare Group A vs Group B results."""
    results_a_path = os.path.join(output_dir, "results_group_A.json")
    results_b_path = os.path.join(output_dir, "results_group_B.json")

    if not os.path.exists(results_a_path) or not os.path.exists(results_b_path):
        logger.error("Need both results_group_A.json and results_group_B.json")
        return

    with open(results_a_path) as f:
        a = json.load(f)
    with open(results_b_path) as f:
        b = json.load(f)

    print("\n" + "=" * 70)
    print("ENVAUDIT TRAINING VALIDATION RESULTS")
    print("=" * 70)

    print(f"\n  {'Metric':<30} {'Group A (ALL)':>15} {'Group B (FILTERED)':>18} {'Delta':>10}")
    print("  " + "-" * 75)
    print(f"  {'Training tasks':<30} {a['train_tasks']:>15} {b['train_tasks']:>18}")
    print(f"  {'Training steps':<30} {a['num_train_steps']:>15} {b['num_train_steps']:>18}")
    print(f"  {'Baseline pass@1':<30} {a['baseline_pass_at_1']:>15.4f} {b['baseline_pass_at_1']:>18.4f}")
    print(f"  {'Trained pass@1':<30} {a['trained_pass_at_1']:>15.4f} {b['trained_pass_at_1']:>18.4f} {b['trained_pass_at_1'] - a['trained_pass_at_1']:>+10.4f}")
    print(f"  {'Improvement':<30} {a['improvement']:>+15.4f} {b['improvement']:>+18.4f}")

    if b["trained_pass_at_1"] >= a["trained_pass_at_1"]:
        print(f"\n  ✓ VALIDATED: Filtered training (Group B) ≥ unfiltered (Group A)")
        print(f"    envaudit filtering improves training by {b['trained_pass_at_1'] - a['trained_pass_at_1']:+.4f} pass@1")
    else:
        print(f"\n  ✗ NOT VALIDATED: Filtered training (Group B) < unfiltered (Group A)")
        print(f"    Delta: {b['trained_pass_at_1'] - a['trained_pass_at_1']:+.4f} pass@1")

    print("=" * 70)


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 7: Colab Training Validation")
    parser.add_argument("--group", choices=["A", "B", "compare"],
                        help="Train group A (all), B (filtered), or compare results")
    parser.add_argument("--data-dir", type=str, default="colab_data",
                        help="Directory with training data")
    parser.add_argument("--output-dir", type=str, default="colab_results",
                        help="Directory for model checkpoints and results")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of GRPO training steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test data loading and reward function without training")
    args = parser.parse_args()

    if args.group == "compare":
        compare_results(args.output_dir)
        return

    if args.dry_run:
        _dry_run(args.data_dir)
        return

    if not args.group:
        parser.error("--group is required (A, B, or compare)")

    os.makedirs(args.output_dir, exist_ok=True)
    results = train_grpo(
        group=args.group,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_train_steps=args.steps,
    )

    print(f"\nGroup {args.group} results:")
    print(f"  Baseline pass@1: {results['baseline_pass_at_1']:.4f}")
    print(f"  Trained pass@1:  {results['trained_pass_at_1']:.4f}")
    print(f"  Improvement:     {results['improvement']:+.4f}")


def _dry_run(data_dir: str):
    """Test data loading and reward function without GPU/training."""
    print("\n=== DRY RUN ===")

    # Test data loading
    for name in ["train_all.jsonl", "train_filtered.jsonl", "eval.jsonl"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            data = load_jsonl(path)
            print(f"  {name}: {len(data)} tasks")
        else:
            print(f"  {name}: NOT FOUND at {path}")

    # Test reward function on a sample
    print("\n  Testing reward function...")
    correct_code = "def add(a, b):\n    return a + b"
    wrong_code = "def add(a, b):\n    return 42"
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]

    r_correct = _execute_and_check(correct_code, tests, [])
    r_wrong = _execute_and_check(wrong_code, tests, [])
    print(f"    Correct code reward: {r_correct} (expected 1.0)")
    print(f"    Wrong code reward:   {r_wrong} (expected 0.0)")

    # Test with weak test (only first assertion)
    r_wrong_weak = _execute_and_check(wrong_code, [tests[0]], [])
    print(f"    Wrong code with weak test: {r_wrong_weak} (demonstrates hackability)")

    print("\n  Dry run complete. Ready for Colab training.")


if __name__ == "__main__":
    main()
