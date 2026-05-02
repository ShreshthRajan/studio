"""
Step 7: Colab Training Validation Script

This script trains Qwen3-4B with GRPO on two datasets and compares results:
  Group A: ALL tasks (includes tasks with weakened tests — simulates hackable env)
  Group B: FILTERED tasks (weak-test tasks removed — envaudit-filtered)

Both groups use identical hyperparameters and compute budget.

Literature-grounded design (see literature_review.md):
  - Per-prompt difficulty filter to pass rate ∈ [0.3, 0.7]: Bae et al. 2025
    (arXiv:2504.03380). Removes prompts with no GRPO advantage signal.
  - GRPOConfig knobs: scale_rewards=False (Liu et al. 2025, arXiv:2503.20783),
    epsilon_high=0.28 + mask_truncated_completions=True (DAPO, arXiv:2503.14476).
  - Constant-with-warmup LR schedule, lr=2e-5: matches GRPO-on-Qwen recipes.
  - Greedy eval (temperature=0.0): removes sampling noise so any LoRA delta
    is detectable.
  - Bootstrap CI + McNemar test: paired statistical comparison of A vs. B.

Usage on Colab:
  !python experiments/colab_training.py --group A --steps 300
  !python experiments/colab_training.py --group B --steps 300
  !python experiments/colab_training.py --group compare

Pre-flight (recommended before --group A on a new dataset):
  !python experiments/colab_training.py --group A --profile-only

Local dry-run (no GPU):
  python experiments/colab_training.py --dry-run
"""

import argparse
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
                   max_new_tokens: int = 512, greedy: bool = True) -> Dict:
    """Evaluate a model on held-out tasks. Default: greedy decoding.

    Greedy (temperature=0, do_sample=False) removes sampling noise so any
    LoRA-induced output change is detectable. Set greedy=False to enable
    sampling-based eval (only useful for pass@k>1, not used in our A/B).
    """
    import torch
    results = []
    total_pass = 0

    for task in eval_tasks:
        prompt = task["prompt"]
        tests = task["test_list"]
        imports = task.get("test_imports", [])

        chat_text = format_chat_prompt(prompt, tokenizer)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        if greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = 0.2

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

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


# ─── Difficulty Profiling (Bae et al. 2025, arXiv:2504.03380) ──────────

def profile_difficulty(model, tokenizer, train_data: List[Dict],
                        n_rollouts: int = 8, temperature: float = 1.0,
                        max_new_tokens: int = 384) -> List[Dict]:
    """Estimate per-prompt pass rate by sampling n_rollouts completions.

    Used to filter the training set to the "learnable middle" — prompts
    where GRPO actually has within-group reward variance. Without this,
    `frac_reward_zero_std` near 1.0 means almost no useful gradient.

    Per-prompt latency: one forward pass with num_return_sequences=n_rollouts
    when VRAM permits, otherwise falls back to a serial loop.
    """
    import torch

    profiles = []
    for i, task in enumerate(train_data, 1):
        prompt = task["prompt"]
        tests = task["test_list"]
        imports = task.get("test_imports", [])

        chat_text = format_chat_prompt(prompt, tokenizer)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        completions: List[str] = []
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=n_rollouts,
                    pad_token_id=tokenizer.eos_token_id,
                )
                completions = [
                    tokenizer.decode(
                        outputs[k][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    for k in range(n_rollouts)
                ]
            except torch.cuda.OutOfMemoryError:
                # Serial fallback for tight VRAM
                torch.cuda.empty_cache()
                for _ in range(n_rollouts):
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    completions.append(tokenizer.decode(
                        out[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    ))

        passes = sum(
            1 for c in completions
            if _execute_and_check(c, tests, imports) == 1.0
        )
        pass_rate = passes / max(n_rollouts, 1)
        profiles.append({
            "task_id": task["task_id"],
            "pass_rate": pass_rate,
            "n_rollouts": n_rollouts,
            "n_pass": passes,
        })

        if i % 25 == 0 or i == len(train_data):
            logger.info("Profiled %d/%d (latest pass_rate=%.2f)",
                        i, len(train_data), pass_rate)

    return profiles


def filter_to_learnable_middle(train_data: List[Dict],
                                profiles: List[Dict],
                                low: float = 0.3,
                                high: float = 0.7) -> List[Dict]:
    """Keep tasks whose per-prompt pass rate lies in [low, high].

    Tasks not present in the profile (e.g., new tasks, profile cache stale)
    are kept by default — better to over-include than to silently drop.
    """
    profile_map = {p["task_id"]: p["pass_rate"] for p in profiles}
    kept = []
    for task in train_data:
        pr = profile_map.get(task["task_id"])
        if pr is None:
            kept.append(task)
            continue
        if low <= pr <= high:
            kept.append(task)
    return kept


def difficulty_distribution(profiles: List[Dict]) -> Dict:
    """Summarize the pass-rate distribution across profiled tasks."""
    if not profiles:
        return {"n": 0}
    rates = [p["pass_rate"] for p in profiles]
    bands = {"0.0": 0, "0.0-0.1": 0, "0.1-0.3": 0, "0.3-0.7": 0,
             "0.7-0.9": 0, "0.9-1.0": 0, "1.0": 0}
    for r in rates:
        if r == 0.0:
            bands["0.0"] += 1
        elif r < 0.1:
            bands["0.0-0.1"] += 1
        elif r < 0.3:
            bands["0.1-0.3"] += 1
        elif r <= 0.7:
            bands["0.3-0.7"] += 1
        elif r <= 0.9:
            bands["0.7-0.9"] += 1
        elif r < 1.0:
            bands["0.9-1.0"] += 1
        else:
            bands["1.0"] += 1
    return {
        "n": len(profiles),
        "mean_pass_rate": round(sum(rates) / len(rates), 4),
        "in_learnable_middle_0.3_0.7": bands["0.3-0.7"],
        "bands": bands,
    }


# ─── Statistics for compare_results ────────────────────────────────────

def _align_pass_vectors(eval_a: Dict, eval_b: Dict) -> Tuple[List[int], List[int], List]:
    """Align Group A and Group B per-task pass/fail by task_id."""
    a_map = {r["task_id"]: bool(r["passed"]) for r in eval_a["per_task"]}
    b_map = {r["task_id"]: bool(r["passed"]) for r in eval_b["per_task"]}
    common_ids = sorted(set(a_map.keys()) & set(b_map.keys()))
    a_vec = [int(a_map[tid]) for tid in common_ids]
    b_vec = [int(b_map[tid]) for tid in common_ids]
    return a_vec, b_vec, common_ids


def bootstrap_diff_ci(a_vec: List[int], b_vec: List[int],
                       n_resamples: int = 10000, alpha: float = 0.05,
                       seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap (mean, lo, hi) of (b - a) on paired binary outcomes.

    Returns (mean_diff, ci_lo, ci_hi) at 1-alpha confidence. The vectors
    must be aligned (same task_id at each index).
    """
    import numpy as np
    if len(a_vec) != len(b_vec):
        raise ValueError("a_vec and b_vec must be the same length (paired).")
    if not a_vec:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    a_arr = np.asarray(a_vec, dtype=float)
    b_arr = np.asarray(b_vec, dtype=float)
    n = len(a_arr)

    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = b_arr[idx].mean() - a_arr[idx].mean()

    lo = float(np.percentile(diffs, 100.0 * alpha / 2.0))
    hi = float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0)))
    mean_diff = float(b_arr.mean() - a_arr.mean())
    return mean_diff, lo, hi


def mcnemar_test(a_vec: List[int], b_vec: List[int]) -> Tuple[float, int, int]:
    """McNemar's exact test on paired binary outcomes.

    Returns (p_value, b_count, c_count) where:
      b = #{a_fail, b_pass} (B improved over A)
      c = #{a_pass, b_fail} (B regressed from A)

    p-value is exact two-sided, computed via the binomial test on
    min(b, c) successes out of (b + c) trials with p=0.5.
    """
    from scipy.stats import binomtest
    if len(a_vec) != len(b_vec):
        raise ValueError("a_vec and b_vec must be the same length (paired).")

    b = sum(1 for x, y in zip(a_vec, b_vec) if not x and y)
    c = sum(1 for x, y in zip(a_vec, b_vec) if x and not y)

    if b + c == 0:
        # No discordant pairs → no evidence either way
        return 1.0, b, c

    result = binomtest(min(b, c), b + c, p=0.5, alternative="two-sided")
    return float(result.pvalue), int(b), int(c)


# ─── GRPOConfig introspection guard ────────────────────────────────────

def _safe_grpo_config(GRPOConfig_cls, **kwargs):
    """Drop kwargs not supported by the installed TRL version, with a warning.

    Defensive against version drift: lets us pass DAPO-era kwargs
    (epsilon_high, mask_truncated_completions, scale_rewards) without
    crashing on older TRL releases that lack them.
    """
    try:
        sig = inspect.signature(GRPOConfig_cls.__init__)
    except (ValueError, TypeError):
        return GRPOConfig_cls(**kwargs)
    valid = set(sig.parameters.keys())
    accepted = {k: v for k, v in kwargs.items() if k in valid}
    rejected = sorted(k for k in kwargs if k not in valid)
    if rejected:
        logger.warning("GRPOConfig does not support these kwargs (skipped): %s",
                       rejected)
    return GRPOConfig_cls(**accepted)


# ─── GRPO Training ─────────────────────────────────────────────────────

def train_grpo(
    group: str,
    data_dir: str,
    output_dir: str,
    num_train_steps: int = 300,
    batch_size: int = 4,
    num_generations: int = 8,
    learning_rate: float = 2e-5,
    max_prompt_length: int = 512,
    max_completion_length: int = 384,
    eval_limit: Optional[int] = None,
    model_name: str = "unsloth/Qwen3-4B",
    profile_only: bool = False,
    profile_n_rollouts: int = 8,
    profile_low: float = 0.3,
    profile_high: float = 0.7,
    skip_profile: bool = False,
):
    """Train a model with GRPO using Unsloth, with literature-validated knobs.

    Defaults set per the salvage plan documented in literature_review.md:
      - num_train_steps=300, num_generations=8, lr=2e-5
      - constant_with_warmup scheduler (warmup_ratio=0.05)
      - scale_rewards=False (Liu et al. 2025)
      - epsilon_high=0.28, mask_truncated_completions=True (DAPO)
      - Difficulty filter to per-prompt pass rate ∈ [profile_low, profile_high]
        (Bae et al. 2025) — profile cached at output_dir/difficulty_profile.json
        and shared between Group A and Group B for fair comparison.
      - Greedy eval (temperature=0).

    profile_only: just compute the difficulty profile and exit (pre-flight).
    skip_profile: bypass difficulty filtering (for debugging / ablation).
    """
    import torch

    if group == "A":
        train_path = os.path.join(data_dir, "train_all.jsonl")
        label = "ALL (includes weak-test tasks)"
    elif group == "B":
        train_path = os.path.join(data_dir, "train_filtered.jsonl")
        label = "FILTERED (weak-test tasks removed)"
    else:
        raise ValueError(f"Group must be 'A' or 'B', got '{group}'")

    train_data_raw = load_jsonl(train_path)
    eval_data = load_jsonl(os.path.join(data_dir, "eval.jsonl"))
    if eval_limit is not None:
        eval_data = eval_data[:eval_limit]
    logger.info("Group %s: %s", group, label)
    logger.info("Train tasks (raw): %d, Eval tasks: %d", len(train_data_raw), len(eval_data))

    # --- Load model with Unsloth ---
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
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

    # --- Difficulty profile (cached, shared across groups) ---
    train_data = train_data_raw
    profile = None
    if not skip_profile:
        os.makedirs(output_dir, exist_ok=True)
        profile_cache = os.path.join(output_dir, "difficulty_profile.json")
        if os.path.exists(profile_cache):
            logger.info("Loading cached difficulty profile from %s", profile_cache)
            with open(profile_cache) as f:
                profile = json.load(f)
        else:
            # Profile the union of A and B (i.e., train_all.jsonl, the superset).
            # We profile against the BASE model (LoRA at zero) so the profile
            # is shared between groups and re-uses cleanly.
            train_all_path = os.path.join(data_dir, "train_all.jsonl")
            if os.path.exists(train_all_path):
                profile_source = load_jsonl(train_all_path)
            else:
                profile_source = train_data_raw
            logger.info("Profiling difficulty on %d tasks "
                        "(n_rollouts=%d, T=1.0)... this is one-time.",
                        len(profile_source), profile_n_rollouts)
            FastLanguageModel.for_inference(model)
            t0 = time.time()
            profile = profile_difficulty(
                model, tokenizer, profile_source,
                n_rollouts=profile_n_rollouts,
            )
            logger.info("Profile complete in %.1f s", time.time() - t0)
            with open(profile_cache, "w") as f:
                json.dump(profile, f, indent=2)
            logger.info("Cached profile to %s", profile_cache)

        dist = difficulty_distribution(profile)
        logger.info("Difficulty distribution: %s", dist)

        if profile_only:
            logger.info("--profile-only set; exiting before training.")
            return {"profile": profile, "distribution": dist}

        before = len(train_data_raw)
        train_data = filter_to_learnable_middle(
            train_data_raw, profile, low=profile_low, high=profile_high,
        )
        logger.info("Difficulty filter [%s, %s]: %d → %d tasks",
                    profile_low, profile_high, before, len(train_data))
        if len(train_data) < 20:
            logger.warning("Filtered training set has only %d tasks. "
                           "Consider relaxing the [%s, %s] band.",
                           len(train_data), profile_low, profile_high)

    if not train_data:
        raise RuntimeError(
            "Training set is empty after filtering. Re-run with --skip-profile "
            "or widen the [profile_low, profile_high] band."
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
            text = completion if isinstance(completion, str) else str(completion)
            rewards.append(_execute_and_check(text, tests, imports))
        return rewards

    # Literature-validated GRPO knobs (see literature_review.md)
    config_kwargs = dict(
        output_dir=os.path.join(output_dir, f"group_{group}"),
        max_steps=num_train_steps,
        per_device_train_batch_size=batch_size,
        num_generations=num_generations,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        logging_steps=10,
        save_steps=num_train_steps,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=42,
        # DAPO clip-higher
        epsilon=0.2,
        epsilon_high=0.28,
        # DAPO: mask truncated completions
        mask_truncated_completions=True,
        # Liu et al. 2025: disable group-std reward scaling
        scale_rewards=False,
    )
    grpo_config = _safe_grpo_config(GRPOConfig, **config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training (Group %s, %d steps, %d train tasks, num_gen=%d)...",
                group, num_train_steps, len(train_data), num_generations)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    logger.info("Training complete in %.1f seconds", train_time)

    # --- Post-training evaluation ---
    logger.info("Evaluating after training...")
    FastLanguageModel.for_inference(model)
    post_eval = evaluate_model(model, tokenizer, eval_data)
    logger.info("Post-training pass@1: %.4f", post_eval["pass_at_1"])

    # --- Save results (including per-task pass vectors for paired stats) ---
    results = {
        "group": group,
        "label": label,
        "model_name": model_name,
        "train_tasks_raw": len(train_data_raw),
        "train_tasks_filtered": len(train_data),
        "eval_tasks": len(eval_data),
        "num_train_steps": num_train_steps,
        "num_generations": num_generations,
        "learning_rate": learning_rate,
        "profile_band": [profile_low, profile_high] if not skip_profile else None,
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

def compare_results(output_dir: str, save: bool = True) -> Dict:
    """Compare Group A vs Group B with paired stats (bootstrap CI + McNemar).

    Aligns per-task pass vectors by task_id and computes:
      - mean Δ pass@1 (B − A) with bootstrap 95% CI
      - McNemar's exact two-sided p-value on discordant pairs
      - Per-task diff: tasks B fixed that A didn't, vice versa
    """
    results_a_path = os.path.join(output_dir, "results_group_A.json")
    results_b_path = os.path.join(output_dir, "results_group_B.json")

    if not os.path.exists(results_a_path) or not os.path.exists(results_b_path):
        logger.error("Need both results_group_A.json and results_group_B.json")
        return {}

    with open(results_a_path) as f:
        a = json.load(f)
    with open(results_b_path) as f:
        b = json.load(f)

    a_vec, b_vec, common_ids = _align_pass_vectors(a["trained_eval"], b["trained_eval"])
    if not common_ids:
        logger.error("No overlapping task_ids between Group A and Group B eval results.")
        return {}

    mean_diff, ci_lo, ci_hi = bootstrap_diff_ci(a_vec, b_vec)
    p_value, b_count, c_count = mcnemar_test(a_vec, b_vec)

    a_pass_set = {tid for tid, p in zip(common_ids, a_vec) if p}
    b_pass_set = {tid for tid, p in zip(common_ids, b_vec) if p}
    b_fixed = sorted(b_pass_set - a_pass_set)
    b_regressed = sorted(a_pass_set - b_pass_set)

    significant = p_value < 0.05 and ci_lo > 0
    summary = {
        "n_eval_tasks": len(common_ids),
        "group_a": {
            "trained_pass_at_1": a["trained_pass_at_1"],
            "improvement": a["improvement"],
            "train_tasks_filtered": a.get("train_tasks_filtered", a.get("train_tasks")),
        },
        "group_b": {
            "trained_pass_at_1": b["trained_pass_at_1"],
            "improvement": b["improvement"],
            "train_tasks_filtered": b.get("train_tasks_filtered", b.get("train_tasks")),
        },
        "delta_pass_at_1": round(mean_diff, 4),
        "ci_95_pct": [round(ci_lo, 4), round(ci_hi, 4)],
        "mcnemar_p_value": round(p_value, 6),
        "mcnemar_b_fail_to_pass": b_count,
        "mcnemar_c_pass_to_fail": c_count,
        "b_fixed_task_ids": b_fixed,
        "b_regressed_task_ids": b_regressed,
        "verdict": (
            "VALIDATED" if significant
            else "INCONCLUSIVE" if ci_lo <= 0 <= ci_hi
            else "NOT VALIDATED"
        ),
    }

    print("\n" + "=" * 76)
    print("ENVAUDIT TRAINING VALIDATION RESULTS")
    print("=" * 76)
    print(f"\n  {'Metric':<32} {'Group A (ALL)':>18} {'Group B (FILTERED)':>20}")
    print("  " + "-" * 72)
    print(f"  {'Trained pass@1':<32} {a['trained_pass_at_1']:>18.4f} {b['trained_pass_at_1']:>20.4f}")
    print(f"  {'Improvement vs baseline':<32} {a['improvement']:>+18.4f} {b['improvement']:>+20.4f}")
    print(f"  {'Train tasks (after filter)':<32} "
          f"{a.get('train_tasks_filtered', a.get('train_tasks', '?')):>18} "
          f"{b.get('train_tasks_filtered', b.get('train_tasks', '?')):>20}")
    print()
    print(f"  Paired statistics on n={len(common_ids)} eval tasks:")
    print(f"    Δ pass@1 (B − A):          {mean_diff:+.4f}")
    print(f"    95% bootstrap CI:           [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"    McNemar p-value (2-sided): {p_value:.4f}")
    print(f"    Discordant pairs:          B fixed {b_count}, B regressed {c_count}")

    if summary["verdict"] == "VALIDATED":
        print(f"\n  ✓ VALIDATED: B > A, p={p_value:.4f}, 95% CI excludes zero.")
        print(f"    envaudit filtering improved training by {mean_diff:+.4f} pass@1.")
    elif summary["verdict"] == "NOT VALIDATED":
        print(f"\n  ✗ NOT VALIDATED: B < A, p={p_value:.4f}.")
        print(f"    Delta: {mean_diff:+.4f}. Filtering hurt — re-examine envaudit thresholds.")
    else:
        print(f"\n  ◇ INCONCLUSIVE: 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] overlaps zero, "
              f"p={p_value:.4f}.")
        print(f"    Effect size Δ={mean_diff:+.4f} too small to distinguish from noise "
              f"at n={len(common_ids)}.")
        print(f"    Consider: aggressive weakening mode, more steps, or larger eval set.")

    print("=" * 76)

    if save:
        comp_path = os.path.join(output_dir, "comparison.json")
        with open(comp_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved comparison summary to %s", comp_path)

    return summary


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 7: Colab Training Validation")
    parser.add_argument("--group", choices=["A", "B", "compare"],
                        help="Train group A (all), B (filtered), or compare results.")
    parser.add_argument("--data-dir", type=str, default="colab_data",
                        help="Directory with training data.")
    parser.add_argument("--output-dir", type=str, default="colab_results",
                        help="Directory for model checkpoints and results.")
    parser.add_argument("--steps", type=int, default=300,
                        help="Number of GRPO training steps (default: 300).")
    parser.add_argument("--num-generations", type=int, default=8,
                        help="GRPO rollouts per prompt (default: 8). DAPO-style.")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="GRPO learning rate (default: 2e-5).")
    parser.add_argument("--model", type=str, default="unsloth/Qwen3-4B",
                        help="Base model. Default: unsloth/Qwen3-4B.")
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="Evaluate on only the first N held-out tasks "
                             "(default: full eval set). Use for fast iteration.")
    parser.add_argument("--profile-only", action="store_true",
                        help="Compute difficulty profile and exit (pre-flight). "
                             "Requires --group A or B to know which dataset.")
    parser.add_argument("--profile-low", type=float, default=0.3,
                        help="Lower bound of learnable-middle pass-rate band.")
    parser.add_argument("--profile-high", type=float, default=0.7,
                        help="Upper bound of learnable-middle pass-rate band.")
    parser.add_argument("--profile-rollouts", type=int, default=8,
                        help="Rollouts per prompt for difficulty profile (default: 8).")
    parser.add_argument("--skip-profile", action="store_true",
                        help="Skip difficulty filter (debug / ablation).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test data loading and reward function without training.")
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
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        eval_limit=args.eval_limit,
        model_name=args.model,
        profile_only=args.profile_only,
        profile_n_rollouts=args.profile_rollouts,
        profile_low=args.profile_low,
        profile_high=args.profile_high,
        skip_profile=args.skip_profile,
    )

    if args.profile_only:
        print("\nProfile-only run complete. See difficulty_profile.json in output dir.")
        dist = results.get("distribution", {})
        if dist:
            print(f"  Tasks profiled:                 {dist.get('n')}")
            print(f"  Mean pass rate:                 {dist.get('mean_pass_rate')}")
            print(f"  Tasks in [0.3, 0.7] band:       "
                  f"{dist.get('in_learnable_middle_0.3_0.7')}")
            print(f"  Pass-rate distribution:         {dist.get('bands')}")
        return

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
