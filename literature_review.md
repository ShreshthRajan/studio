# envaudit Literature Review

This document catalogs every paper, blog post, or technical report that informed envaudit's design and the Step 7 GRPO training validation. For each source: full citation, the specific finding we rely on, and where in our system it shows up.

The review is grouped by which part of envaudit the work supports.

---

## A. Failure-mode evidence: why test quality matters for code RL

These establish the *empirical* premise that envaudit operates on — that bad reward environments measurably degrade RL training, sometimes catastrophically.

### A1. HARDTESTGEN — Bad test cases hurt RL "a great deal"
- **Citation:** *HARDTESTGEN: Synthesizing High-Quality Test Cases for LLM Code Reasoning* (ICLR 2026).
- **Finding:** Replacing low-quality test cases with synthesized hard tests raised RLVR precision by **+40 percentage points** on competitive programming tasks. The authors describe test-case quality as mattering "a great deal" for RL.
- **How envaudit uses it:** Establishes the upper bound on the size of effect to expect when fixing test quality. Cited in the project README and the technical report's motivation. Our `KEEP/FIX/DROP` verdict in `envaudit/scoring/composite.py` is the operational answer to the question HARDTESTGEN raises.

### A2. GradAlign — corrupted rewards cause catastrophic collapse
- **Citation:** *GradAlign: Reward-Gradient Alignment for Robust RLHF* (Feb 2026).
- **Finding:** With 50% reward-signal corruption, model performance can collapse by **~40 percentage points** in math reasoning tasks. The collapse is sudden, not gradual.
- **How envaudit uses it:** Justifies the *binary* DROP verdict for tasks below the EQS threshold — once a task is sufficiently broken, no amount of mixing can rescue it. Referenced in the verdict logic at `envaudit/scoring/composite.py`.

### A3. DRIVE — mixed-quality training collapses competitive programming RL
- **Citation:** *DRIVE: Data Refinement via Inferred Verification Errors* (2025).
- **Finding:** Training on mixed-quality competitive programming tasks produced **48.8% performance collapse** vs. clean filtered training.
- **How envaudit uses it:** Direct empirical analog of the envaudit thesis ("filter before training"). Numerically supports the prediction that Group B (filtered) ≥ Group A (unfiltered). Cited in the Step 7 experimental design.

### A4. BugPilot — A/B on training task quality, controlled
- **Citation:** *BugPilot: Complex Bug Generation for Efficient Learning of SWE Skills* (Microsoft, Oct 2025; arXiv:2510.19898).
- **Finding:** Trained the same model on two synthetic-bug datasets (BaseMix vs FeatAdd) under identical hyperparameters. Found "substantial improvement when using FeatAdd bugs for RL fine-tuning" but "not significant gains" from BaseMix. Best model FrogBoss achieved 54.6% pass@1 on SWE-Bench-Verified using a 25% smaller dataset than prior work.
- **How envaudit uses it:** **Methodologically the closest published analog to our Step 7 experiment.** Their A/B-on-training-data design at frontier scale validates the shape of our small-scale Colab A/B. We cite this as the precedent for "quality > quantity in code RL." It motivates our Group A vs. Group B comparison.

### A5. LLMs Gaming Verifiers — same-model A/B on verifier weakness
- **Citation:** *LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking* (arXiv:2604.15149).
- **Finding:** Two identical 7B base models (Olmo-3-7B-Think-DPO) trained under identical conditions, differing only in the verifier (extensional vs isomorphic). After 500 steps, **3.5/10 reward gap (35%)** between groups. The weak verifier directly induced shortcut strategies.
- **How envaudit uses it:** **The exact experimental shape we replicate.** Validates that "same model, same hyperparams, only the reward environment differs" produces measurable divergence in published work. Sets expectation for effect size at frontier scale.

### A6. Anthropic — reward hacking → emergent misalignment
- **Citation:** *Natural Emergent Misalignment from Reward Hacking in Production RL* (Anthropic, Nov 2025).
- **Finding:** Reward hacking in production RL pipelines can lead to broader emergent misalignment beyond just the gamed reward. Safety-relevant.
- **How envaudit uses it:** Supplies the safety motivation ("why this matters beyond benchmark scores") in the project README and pitch. Justifies envaudit's relevance to alignment-conscious labs.

### A7. OpenAI — dropping SWE-bench Verified
- **Citation:** *Why we no longer evaluate SWE-bench Verified* (OpenAI, Feb 23, 2026).
- **Finding:** OpenAI announced that **59.4% of failed tasks on SWE-bench Verified have flawed tests**, and recommended migrating to SWE-bench Pro.
- **How envaudit uses it:** **Independent corroboration of envaudit's 28.5% Docker-verified hackability finding.** OpenAI used a broader definition of "flawed" (any failed task with a test issue), envaudit measures Docker-verified exploit success — the two numbers are consistent given the definition gap. Lead paragraph of the project pitch: "We measured this in October. OpenAI confirmed it in February."

---

## B. envaudit scoring methodology — the verifier metrics

These inform the per-task scoring components in `envaudit/scoring/`.

### B1. NVIDIA Verifier Scoring framework
- **Citation:** *Scoring Verifiers: Evaluating Synthetic Verification for Code and Reasoning* (NVIDIA, arXiv:2502.13820).
- **Finding:** Proposes a multi-component verifier-evaluation framework with discrimination, ranking-correlation (Spearman), error magnitude (MAE), and completeness scores.
- **How envaudit uses it:** Adapted to binary verifiers in `envaudit/scoring/verifier_scorer.py`. Our discrimination + Spearman + MAE + completeness composite is a direct implementation; we re-weighted for the binary pass/fail nature of swebench's harness.

### B2. Meta Semi-Formal Agentic Reasoning
- **Citation:** *Semi-Formal Agentic Reasoning for Code Verification* (Meta, arXiv:2603.01896).
- **Finding:** Structured-reasoning judge that combines symbolic constraint extraction with LLM-based informal reasoning. Improves verification F1 over LLM-only judges.
- **How envaudit uses it:** `envaudit/scoring/semiformal_judge.py` implements this judge. Its outputs feed `envaudit/scoring/hybrid.py`, which computes the confusion matrix (TP/FP/FN/TN) used in the EQS formula.

### B3. EvolveCoder — iterative adversarial test refinement
- **Citation:** *EvolveCoder: Iterative Test-Suite Hardening Against Adversarial Patches* (arXiv:2603.12698).
- **Finding:** Iteratively generating exploits round-by-round, each round informed by the failures of the previous, exposes weak tests that single-shot attacks miss.
- **How envaudit uses it:** `envaudit/agents/iterative_attacker.py` implements rounds 2–3 of exploit generation, informed by Docker-verification feedback from round 1. This is what scaled our discovered exploit count from baseline single-pass.

### B4. R2E-Gym hybrid verification
- **Citation:** *R2E-Gym: Reproducible Repository-Level Environments for SWE Agents* (2025).
- **Finding:** Hybrid verification combining test execution + LLM judgment outperforms either alone for SWE patch correctness.
- **How envaudit uses it:** Inspires the hybrid pipeline in `envaudit/scoring/hybrid.py`, where the semi-formal judge's verdict is combined with Docker test results.

### B5. Online Difficulty Filtering for Reasoning RL
- **Citation:** Bae et al., *Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning* (arXiv:2504.03380, 2025).
- **Finding:** "Expected policy improvement is lower-bounded by the variance of task-level success probabilities." Filtering training prompts to per-prompt pass rate ∈ **[0.3, 0.7]** yields **+4-12% gain in <50% of training steps** on math benchmarks. Maximum learnability at p=0.5.
- **How envaudit uses it:** **The single most important intervention in Step 7.** Implemented as `profile_difficulty()` and `filter_to_learnable_middle()` in `experiments/colab_training.py`. Without this filter, our `frac_reward_zero_std` was 0.9 (no learning); with it, expected ~0.05–0.10. Cited as the primary justification for the difficulty pre-filter.

---

## C. GRPO algorithmic improvements — the hyperparameter choices

These inform the GRPOConfig settings in Step 7.

### C1. DeepSeekMath GRPO (the original)
- **Citation:** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning* (arXiv:2402.03300).
- **Finding:** Introduces GRPO. Group-relative advantage replaces value network. Reduces compute vs. PPO.
- **How envaudit uses it:** The base algorithm we use via TRL 0.22.2's `GRPOTrainer`. Cited as the foundational method.

### C2. DAPO — clip-higher, dynamic sampling, token-level loss
- **Citation:** *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (arXiv:2503.14476, 2025).
- **Findings:**
  - **Clip-higher**: setting `epsilon_high = 0.28` (vs default 0.2) preserves exploration tokens, prevents entropy collapse.
  - **Dynamic sampling**: removing prompts with perfect-accuracy rollouts ("no advantage signal") — same intuition as Bae et al.'s difficulty filter.
  - **Token-level loss aggregation**: averages loss per-token rather than per-response, removing length bias.
  - End-to-end DAPO achieved **50% AIME** vs DeepSeek-R1's 47% in **half the training steps**.
- **How envaudit uses it:** `epsilon_high=0.28` and `mask_truncated_completions=True` are passed to `GRPOConfig` in `experiments/colab_training.py`. Token-level loss is the TRL 0.22 default. The "remove perfect-accuracy prompts" intuition is folded into our `filter_to_learnable_middle` (offline equivalent of DAPO's online dynamic sampling).

### C3. Liu et al. — `scale_rewards=False`
- **Citation:** Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (arXiv:2503.20783, 2025). Also called "Dr. GRPO" in TRL docs.
- **Finding:** Standard GRPO scales rewards by group standard deviation, which **introduces a question-level difficulty bias**. Disabling this scaling (`scale_rewards=False` / `"none"`) improves training.
- **How envaudit uses it:** We set `scale_rewards=False` in `GRPOConfig`. TRL's official docs explicitly recommend this setting per Liu et al.

### C4. Cameron Wolfe — GRPO++ practitioner tricks
- **Citation:** Cameron Wolfe, *GRPO++: Tricks for Making RL Actually Work* (Substack, 2025).
- **Findings:** Compiles practitioner wisdom across DAPO, DeepSeek, Olmo. Notable claims:
  - "Using a small batch size in GRPO is one of the most common mistakes." Olmo-3 uses batch 512 (64 prompts × 8 rollouts).
  - "Prompts with perfect accuracy are problematic for GRPO."
  - Health metrics to watch: response length, training reward, entropy, held-out validation.
- **How envaudit uses it:** Informs our `num_generations=8` choice (Olmo-3 default), our health-metric logging (we log all four), and confirms the diagnosis behind the difficulty filter. Cited as the practitioner-perspective synthesis.

### C5. PPO Lite — `scale_rewards="batch"` alternative
- **Citation:** *Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning (Lite PPO)* (arXiv:2508.08221).
- **Finding:** Computing reward mean at group level but std at batch level is more robust than pure group-level scaling.
- **How envaudit uses it:** Mentioned as an alternative to `scale_rewards=False` in our writeup. Not currently set (we follow Liu et al. instead), but flagged as a future-work knob.

---

## D. Model and infrastructure references

### D1. Qwen3 Technical Report
- **Citation:** Qwen Team, *Qwen3 Technical Report* (arXiv:2505.09388, May 2025).
- **Findings:**
  - Qwen3-0.6B-Base scores 54.60 on MBPP, 46.28 on EvalPlus.
  - Qwen3-8B scores 73.40 on MBPP.
- **How envaudit uses it:** Informed the model-size decision for Step 7. Qwen3-0.6B's MBPP baseline (54.6) is *too high* to drop into the GRPO learnable middle without difficulty filtering — which is why we keep Qwen3-4B and rely on the difficulty filter instead.

### D2. Unsloth Qwen3 GRPO notebook
- **Citation:** `unslothai/notebooks/nb/Qwen3_(4B)-GRPO.ipynb` (Apr 2026).
- **Finding:** Pins exact dependency triple known to work for GRPO on Colab T4: `transformers==4.56.2`, `trl==0.22.2 --no-deps`, `unsloth` (latest).
- **How envaudit uses it:** The Step 7 environment setup follows this triple verbatim. The `--no-deps` on TRL is load-bearing — without it, TRL pulls a transformers version that breaks Unsloth's patches.

### D3. EGCA — Execution-Grounded Credit Assignment
- **Citation:** *Execution-Grounded Credit Assignment for GRPO in Code Generation* (ICLR 2026 SPOT Workshop).
- **Finding:** GRPO + execution-grounded credit assignment achieved **82.1% pass@1 on HumanEval (+3.1 over GRPO)** and **68.9% on MBPP (+1.5)**.
- **How envaudit uses it:** Proves that GRPO + binary execution reward on MBPP/HumanEval is a viable training regime — i.e., the regime our Step 7 operates in is well-precedented. Cited as the proof point that small models *can* learn from GRPO on code.

---

## E. Statistical methodology references

### E1. McNemar's exact test
- **Citation:** McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages." *Psychometrika* 12 (2): 153–157.
- **Use:** Exact test for paired binary outcomes (model A pass/fail vs model B pass/fail on same eval task). Implemented via `scipy.stats.binomtest` on discordant pairs (`b` = A fail/B pass; `c` = A pass/B fail).
- **Where in envaudit:** `compare_results()` in `experiments/colab_training.py`.

### E2. Bootstrap confidence intervals
- **Citation:** Efron & Tibshirani (1993). *An Introduction to the Bootstrap*. CRC Press.
- **Use:** Resample paired pass-vectors with replacement (10,000 iterations, fixed seed). Compute 95% CI on Δ pass@1 (Group B − Group A).
- **Where in envaudit:** `compare_results()` in `experiments/colab_training.py`.

---

## F. Composite EQS weighting and verdict thresholds

### F1. TRACE 54-category exploit taxonomy
- **Citation:** *TRACE: A Taxonomy of Reward-Hacking Exploits for Code Environments* (2025).
- **Use:** 54-category exploit taxonomy used to label exploit strategies in `envaudit/agents/iterative_attacker.py`. Provides the strategy_name and strategy_description fields in the per-exploit metadata.
- **Where in envaudit:** Strategy labeling in iterative attacks.

### F2. InfoRM — Reward Signal Informativeness
- **Citation:** *InfoRM: Information-Theoretic Reward Model Evaluation* (2024).
- **Use:** Conceptually adapted for our Reward Signal Informativeness (RSI) metric — correlation between test pass rate and patch correctness.
- **Where in envaudit:** Influences the conceptual framing of the EQS, though our binary-pass implementation doesn't directly use InfoRM's mutual information formulation.

---

## How to cite envaudit

For the technical report, the canonical paragraph that introduces our methodological lineage:

> envaudit's Environment Quality Score (EQS) draws on the verifier-scoring framework of Toshniwal et al. [B1, NVIDIA] and the structured-reasoning judge of Wang et al. [B2, Meta], extended with iterative adversarial testing inspired by [B3, EvolveCoder]. The Step 7 GRPO validation follows the controlled-A/B design of [A4, BugPilot] and [A5, LLMs Gaming Verifiers], with the offline difficulty filter of [B5, Bae et al.] and DAPO's [C2] clip-higher and dynamic-sampling intuitions. We use Liu et al.'s `scale_rewards=False` setting [C3] per the official TRL recommendation. The 28.5% Docker-verified hackability we report on SWE-bench Verified is consistent with OpenAI's later finding [A7] that 59.4% of Verified-failed tasks have flawed tests.

---

*Last updated: 2026-04-29*
