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
import re
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


def _extract_code(text: str) -> str:
    """Extract executable Python code from model output.

    Handles:
      - Qwen3 thinking traces: <think>...</think> (stripped)
      - Markdown code fences: ```python ... ``` (extracted)
      - Prose prefix before a def/import/class (trimmed)
      - Trailing prose after code (trimmed via ast.parse back-off)
    """
    if not text:
        return ""

    # Strip Qwen3 reasoning/thinking traces
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also strip any unclosed think blocks at start
    text = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL)

    # Prefer the first fenced Python block if present
    fence_match = re.search(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # No fence: strip common prose prefixes by finding the first code token.
    code_start = re.search(r'^(def |import |from |class |@)', text, re.MULTILINE)
    if code_start:
        text = text[code_start.start():]

    text = text.strip()
    if not text:
        return ""

    # Strip trailing prose via ast back-off: pop one line at a time from
    # the end until the remaining text parses as valid Python (or we run
    # out of lines). This handles "def foo():\n    ...\n\nThat's it!"
    import ast
    lines = text.split("\n")
    while lines:
        candidate = "\n".join(lines)
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            lines.pop()
    return text


def _execute_and_check(code: str, tests: List[str], imports: List[str],
                       timeout: int = 5) -> float:
    """Execute code + tests in a subprocess, return 1.0 if all pass.

    The `code` input may be raw model output (with think tags, markdown
    fences, or prose); _extract_code handles the cleanup.
    """
    extracted = _extract_code(code)
    # Build the test script
    import_block = "\n".join(imports) if imports else ""
    test_block = "\n".join(tests)
    script = f"{import_block}\n{extracted}\n{test_block}"

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


# ─── Chat Formatting ───────────────────────────────────────────────────

def format_chat_prompt(prompt_text: str, tokenizer) -> str:
    """Apply the tokenizer's chat template to a user prompt.

    Qwen3 (and most modern instruct models) require chat-formatted input
    with role tags. Raw text prompts produce degenerate completions.

    Disables Qwen3's reasoning/thinking mode when supported, so the model
    emits code directly instead of long <think>...</think> traces that
    consume the generation budget.
    """
    messages = [{"role": "user", "content": prompt_text}]
    # Qwen3 tokenizers accept enable_thinking; older tokenizers do not.
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# ─── Evaluation ────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, eval_tasks: List[Dict],
                   max_new_tokens: int = 512, n_samples: int = 1) -> Dict:
    """Evaluate a trained model on held-out tasks.

    Generates code for each task (using the chat template), extracts
    executable code from the output, and executes against full test
    suites. Computes pass@1.
    """
    import torch
    results = []
    total_pass = 0

    for task in eval_tasks:
        prompt = task["prompt"]
        tests = task["test_list"]
        imports = task.get("test_imports", [])

        # Apply chat template so the model responds as an assistant.
        chat_text = format_chat_prompt(prompt, tokenizer)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # _execute_and_check handles code extraction internally.
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
    eval_limit: Optional[int] = None,
):
    """Train Qwen3-4B with GRPO using Unsloth.

    Args:
        group: "A" (all tasks) or "B" (filtered tasks)
        data_dir: Directory containing train_all.jsonl, train_filtered.jsonl, eval.jsonl
        output_dir: Where to save the trained model and results
        num_train_steps: Number of GRPO training steps
        eval_limit: If set, evaluate on only the first N held-out tasks.
                    Useful for fast iteration; defaults to full eval set.
    """
    import torch

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
    if eval_limit is not None:
        eval_data = eval_data[:eval_limit]
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

    # Format training data for GRPO.
    # We chat-template the prompts here so the model sees properly-formatted
    # user turns during rollouts. Tests are keyed by the chat-templated
    # prompt so the reward function can look them up from what GRPO passes
    # back.
    train_formatted = format_for_grpo(train_data)
    chat_prompts = [format_chat_prompt(t["prompt"], tokenizer) for t in train_formatted]
    train_dataset = Dataset.from_list([{"prompt": cp} for cp in chat_prompts])

    prompt_to_tests: Dict[str, tuple] = {}
    for cp, t in zip(chat_prompts, train_formatted):
        prompt_to_tests[cp] = (t["test_list"], t.get("test_imports", []))

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            tests, imports = prompt_to_tests.get(prompt, ([], []))
            # Completions are strings; _execute_and_check handles extraction
            # of code from markdown fences / thinking traces.
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
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
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
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Evaluate on only the first N held-out tasks "
                             "(default: full eval set). Use for fast iteration.")
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
        eval_limit=args.eval_limit,
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

    # Test with weak test (only first assertion that the wrong code happens to satisfy)
    hackable_code = "def add(a, b):\n    return 3"  # hardcodes first test's answer
    r_hack_weak = _execute_and_check(hackable_code, [tests[0]], [])
    r_hack_full = _execute_and_check(hackable_code, tests, [])
    print(f"    Hackable code with weak test:  {r_hack_weak} (demonstrates hackability)")
    print(f"    Hackable code with full tests: {r_hack_full} (should be 0.0)")

    # Test the code extractor on realistic model output
    print("\n  Testing code extraction (handles think tags + markdown fences)...")
    model_output = (
        "<think>Let me solve this step by step. I need to add two numbers.</think>\n"
        "Sure! Here's the solution:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```"
    )
    r_extracted = _execute_and_check(model_output, tests, [])
    print(f"    Model output with think+fence reward: {r_extracted} (expected 1.0)")

    print("\n  Dry run complete. Ready for Colab training.")


if __name__ == "__main__":
    main()
