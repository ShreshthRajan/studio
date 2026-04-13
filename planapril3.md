# PLAN: Predicting RL Training Failures from Environment Quality

## Context: What we're actually trying to solve

**Original problem (OAI RL lead):** "Create an eval for RL envs such that when they run small runs they can easily hill climb on it to optimize the env quickly, then run big run."

**What this means:** Before spending $1M on RL training, labs need to know which tasks in their training set will produce good learning signal vs. garbage.

**What we built so far:** A system that generates exploit patches against SWE-bench test suites and Docker-verifies them. Found 28.5% of SWE-bench Verified is hackable. Cost: $1.50.

**Why that's not enough:** We proved tests are weak. We did NOT prove that weak tests → bad RL training. That causal link is the entire ball game. Without it, we're just confirming what SWE-ABS, UTBoost, and SWE-bench+ already showed from different angles.

## The Key Research Findings

From deep research across ~30 papers and industry sources:

### The causal link is partially established but NOT proven for code RL:

| Evidence | Source | Domain |
|----------|--------|--------|
| Bad test cases matter "a great deal" for RL; +40pp precision | HARDTESTGEN (ICLR 2026) | Competitive programming |
| 50% corrupted rewards → 40pp catastrophic collapse | GradAlign (Feb 2026) | Math |
| Training on mixed-quality tasks → 48.8% performance collapse | DRIVE (2025) | Competitive programming |
| SWE-Smith fails for RL, R2E-Gym works (quality > quantity) | BugPilot (Microsoft, Oct 2025) | SWE-bench |
| Reward hacking → emergent misalignment (safety) | Anthropic (Nov 2025) | Production code |
| **Removing broken tasks improves code RL on SWE benchmarks** | **NOBODY** | **This is the gap** |

### Critical complication — "Spurious Rewards" finding:
Random rewards improve Qwen2.5-Math-7B by 21.4pp vs. ground truth's 29.1pp (73% of real benefit). For some model/task combos, RL mostly amplifies pretraining priors — reward quality barely matters. BUT: model-dependent (Qwen yes, Llama/OLMo no) and works less for tasks requiring genuine new capability.

### DeepSWE is fully open-source:
Together AI published EVERYTHING — model, 4.5K R2E-Gym tasks, training code (rLLM), W&B training logs, eval logs. Per-task training metrics should be extractable. Nobody has analyzed them for reward hacking patterns.

### OpenAI dropped SWE-bench Verified (Feb 23, 2026):
Found 59.4% of failed tasks have flawed tests. Recommended moving to SWE-bench Pro. This validates our finding (28.5% hackable) and proves the need for environment quality tooling.

## The 9+ Approach: Mutation-Based Environment Scoring + Causal Validation

### Why this is different from what we had

Old approach: "Use Claude to generate exploit patches" → technically shallow, SWE-ABS already does better.

New approach: **Use mutation testing (a principled SE technique) to compute a quantitative test-suite quality score, then validate that score predicts actual RL training outcomes using DeepSWE's open-source training logs.**

This bridges software engineering and RL — a genuinely novel cross-disciplinary contribution that goes beyond "call an LLM."

### The three metrics (MII Triad):

**Metric 1: Mutation Kill Rate (MKR)**
- Generate N code mutations of the gold patch (off-by-one errors, boundary changes, operator swaps, wrong variable names)
- Run the test suite against each mutation
- MKR = fraction of mutations detected by tests
- Low MKR = test suite can't distinguish correct code from subtly-wrong code = gameable reward
- This is cheap, deterministic, and well-grounded in SE theory (decades of mutation testing research)
- **Never applied to RL environment quality**

**Metric 2: LLM Exploit Success Rate (LESR)**
- What we already have: Claude generates exploit patches, Docker verifies them
- LESR = fraction of high-confidence exploits that pass Docker
- This captures "can a smart agent specifically game this?" which mutation testing misses
- Complements MKR: mutation testing catches random noise sensitivity, LESR catches deliberate gaming

**Metric 3: Reward Signal Informativeness (RSI)**
- Sample K candidate patches of varying quality (gold patch, LLM exploits, random mutations, empty patch)
- Compute correlation between test pass rate and patch correctness
- RSI = how well the reward signal (pass/fail) tracks actual solution quality
- Adapts the InfoRM framework from reward models to environments

### The causal validation (what makes this a 9+):

**Step 1:** Score DeepSWE's 4.5K R2E-Gym training tasks with our metrics
**Step 2:** Extract per-task training metrics from DeepSWE's W&B logs (reward trajectory, solve rate over training steps, whether reward plateaus or spikes suspiciously)
**Step 3:** Compute correlation: do our quality scores predict which tasks DeepSWE's model learned genuinely vs. reward-hacked?
**Step 4:** If yes → first evidence that pre-training environment quality scores predict RL training outcomes

This is the experiment nobody has run. The data exists (DeepSWE is open). The metrics are principled. The result would be genuinely publishable.

### What this gives the OAI lead:

```
envaudit score --tasks my_training_set.jsonl

Task                    MKR    LESR   RSI    Verdict
django-11276           0.23   0.67   0.31   DROP (gameable)
django-11133           0.31   1.00   0.28   DROP (gameable)
astropy-7166           0.15   0.33   0.45   FIX (augment tests)
django-11451           0.89   0.00   0.92   KEEP (robust)
...

Summary: 847/4500 tasks flagged. Recommended action:
- DROP 312 (hopelessly gameable, MKR < 0.2)
- FIX 535 (augmentable, 0.2 < MKR < 0.6)
- KEEP 3653 (robust, MKR > 0.6)
```

## Implementation Plan

### Phase 1: Mutation Testing Engine (3-4 days)
**Files to create/modify:**
- `envaudit/mutation/engine.py` — Mutation operator library (AST-based: off-by-one, boundary, operator swap, variable rename, statement delete, return value change)
- `envaudit/mutation/scorer.py` — Runs mutations against test suite, computes MKR
- `.github/workflows/compute_mkr.yml` — GH Actions workflow for Docker-based mutation testing

**Technical approach:**
- Parse gold patch to extract changed functions
- Apply standard mutation operators (ast.NodeTransformer)
- For each mutant: create a "mutant patch" (gold patch with one mutation)
- Run through swebench Docker harness (same as exploit verification)
- MKR = mutants_killed / total_mutants

**Why GH Actions:** Mutation testing requires running tests in Docker. Same infra we already have.

### Phase 2: Compute scores for SWE-bench Verified (1-2 days)
- Run MKR on all 500 Verified tasks via GH Actions (parallelizable across multiple workflow runs)
- Run LESR using existing exploit pipeline (scale to 500 tasks, ~$15 Claude API)
- Combine into composite Environment Quality Score (EQS)

### Phase 3: DeepSWE Validation (2-3 days)
**Files to create:**
- `experiments/deepswe_validation.py` — Downloads DeepSWE task list, correlates with our scores
- `experiments/wandb_analysis.py` — Extracts per-task metrics from DeepSWE W&B logs

**Steps:**
1. Get DeepSWE's 4.5K task list from their HuggingFace dataset
2. Score each task (MKR via GH Actions, LESR via Claude API)
3. Download W&B training logs via wandb API
4. Extract per-task metrics: reward over training steps, solve rate, reward variance
5. Compute correlations: EQS vs. training metrics
6. Report: "Tasks with low EQS showed X% higher reward variance / Y% more reward collapse during DeepSWE training"

**Risk:** W&B logs may only have aggregate metrics, not per-task. Mitigation: check the rLLM training code to see what's logged.

### Phase 4: Paper (3-5 days, target NeurIPS 2026 ~May deadline)
**Title:** "Predicting Reward Hacking in Code RL: Mutation-Based Environment Quality Scoring"

**Contributions:**
1. MII Triad: three principled pre-training metrics for RL environment quality
2. First mutation-testing-based approach to RL environment assessment (SE × RL bridge)
3. 28.5% Docker-verified hackability rate on SWE-bench Verified (confirms OpenAI's findings)
4. Causal validation: EQS predicts training outcomes on DeepSWE's 4.5K tasks
5. Open-source tool: `envaudit`

**Why this is publishable:**
- Novel method (mutation testing for RL, never done)
- Causal validation (first correlation with actual training logs)
- Timely (OpenAI just dropped SWE-bench Verified, Anthropic showed reward hacking → misalignment)
- Practical (working tool that labs can use)

## Risks and Honest Assessment

### What could go wrong:
1. **DeepSWE W&B logs don't have per-task data** → We can't do the causal validation. Mitigation: check the rLLM source code first. If per-task data isn't there, we'd need to use aggregate metrics or find another open-source RL run.

2. **Mutation testing doesn't correlate with actual hackability** → Our principled metric doesn't predict what matters. Mitigation: We already have LESR (LLM exploits) as a second metric. And SWE-ABS showed mutation-style testing works (their "mutation-driven adversarial testing" is conceptually similar).

3. **Spurious Rewards problem** → For some model/task combos, reward quality doesn't matter. Our scores would predict something that doesn't actually affect training. Counter: this is model-dependent, and the tasks where it fails are pretraining-dominated (easy). For genuinely hard tasks (the ones that matter for frontier capability), reward quality should matter more.

4. **NeurIPS deadline might be tight** → May 2026 is ~4-5 weeks away. We'd need all experiments done in 2-3 weeks. Tight but feasible if Phase 1-3 go smoothly. Alternative: ICML 2026 workshop (July), or NeurIPS 2026 workshop track.

5. **Mutation testing at scale requires a LOT of Docker runs** → N mutations × M tasks. For 500 tasks × 10 mutations = 5,000 Docker runs. At ~2 min each = ~170 hours of GH Actions compute. Free tier = 2,000 min/month for private repos. Mitigation: use public repo (unlimited), or parallelize across multiple workflow runs, or test fewer mutations per task.

### What makes this a 9+:
- Technically novel (mutation testing for RL is new, SE × RL bridge)
- Principled (not "call Claude and see what happens")
- Validates causally (correlates with real training logs)
- Practically useful (labs can run this today)
- Timely (OpenAI dropped SWE-bench Verified, Anthropic showed reward hacking → misalignment)
- Feasible with our constraints (Claude API + free GH Actions, no GPU needed)

### What keeps it below 9:
- We haven't verified DeepSWE W&B logs have per-task data (the whole validation hinges on this)
- Mutation testing is computationally expensive at scale
- The "spurious rewards" finding could undermine the entire premise
- We're still not RUNNING RL training ourselves — correlating with someone else's logs is weaker than running our own ablation

## VALIDATION RESULTS (April 3, 2026)

### Validation 1: Does DeepSWE log per-task data?

**RESULT: NO — W&B logs are aggregate only.**

What we found by reading the rLLM source code:
- W&B logs contain only batch-level aggregates: `batch/solve_none`, `batch/solve_all`, `critic/full-score/{mean|max|min}`
- UIDs are ephemeral random UUIDs per batch — NOT instance_ids. Thrown away after each batch.
- The R2E-Gym-Subset dataset (4,578 tasks) has NO `instance_id` column. Closest identifier is `docker_image`.
- Rewards are binary: 1.0 (passes tests) or 0.0 (fails/timeout).

**Partial workarounds exist but are fragile:**
- `chat_completions/{step}.jsonl` files contain per-trajectory data with `problem_statement` — could reverse-match to instances
- A community dataset (AxT-dev/qwen3-32b-rl-step200-r2e-gym-trajectories) has per-episode data with reward, exit_reason, problem_statement
- Eval logs on Google Drive likely have per-task results

**Bottom line:** Direct causal validation via W&B is NOT feasible without significant reverse-engineering. The community trajectory dataset is the best fallback but covers a different model checkpoint.

### Validation 2: Does mutation testing work on SWE-bench tasks?

**RESULT: TECHNICALLY FEASIBLE but WEAKLY CORRELATED with hackability.**

What we found:
- **Mutations CAN be generated from diffs alone** — no source checkout needed. Text-level mutations (operator swap, literal change, statement deletion) on added lines produce valid unified diffs.
- **Average ~8-15 meaningful mutations per task** across our 50 tasks (mean 10.3 mutable added lines).
- **Existing Docker infrastructure handles it** — same swebench harness, same GH Actions workflow.
- **49/50 tasks have added lines to mutate** (1 pure-deletion needs special handling).

BUT the critical finding:
- **MKR measures a different dimension than hackability.** MKR = "how precisely do tests pin down the exact gold patch semantics." Hackability = "can an LLM produce a structurally different fake that fools the tests."
- Hackable exploits are often **not mutations of the gold patch** — they're hardcoded outputs, format-only fixes, or stack-inspecting hacks. MKR won't capture these.
- Tasks with 700+ regression tests (astropy-13236, astropy-14369) are hackable despite presumably high MKR — because the FAIL_TO_PASS tests are narrow even if PASS_TO_PASS tests are many.
- **MKR is a useful complementary signal but NOT a replacement for LLM exploit testing.**

### Decision Matrix Result

| Validation | Result | Implication |
|------------|--------|-------------|
| DeepSWE per-task logs | NO (aggregate only) | Causal validation via W&B not directly feasible |
| Mutation testing feasible | YES (technically) | Can compute MKR but weakly predicts hackability |
| MKR correlates with hackability | LIKELY WEAK | Different dimension — complement, not substitute |

**We're in the "Neither works cleanly" quadrant.** Neither validation opened a clear 9+ path.

### What this means for the project

The honest situation:
1. **Causal validation is out of reach** without running our own RL training (which requires GPU compute we don't have) or significant reverse-engineering of DeepSWE's trajectory data.
2. **Mutation testing is a nice-to-have metric** but won't carry the project on its own. It measures test precision, not hackability.
3. **Our LLM exploit generation + Docker verification remains our strongest unique asset** — 14/49 Docker-verified hackable tasks at $1.50 cost.

### Revised path forward

Given these findings, the most honest and useful directions are:

**Option A: Ship the tool (practical value, no academic pretension)**
Package what works: LLM exploit generation + Docker verification + mutation testing as complementary metric. Make it easy for labs to run: `envaudit scan --dataset my_tasks.jsonl`. The value is practical — "run this before training, get a report." Don't claim causal validation we can't prove.

**Option B: Pursue causal validation via community trajectory data**
The AxT-dev trajectory dataset has per-episode reward data from a Qwen3-32B model trained on R2E-Gym for 200 steps. We could:
1. Score those tasks with our metrics
2. Correlate with per-episode rewards from the trajectory data
3. This is weaker than DeepSWE W&B logs but still novel
Risk: matching problems between our task scoring and their trajectory data format.

**Option C: Focus on the OpenAI correlation**
OpenAI audited 27.6% of SWE-bench Verified and found 59.4% flawed. We found 28.5% hackable on our 49-task sample. Do our flagged tasks overlap with theirs? If we can get OpenAI's specific flagged task list, this is a strong validation without needing RL training data.

**Recommendation: A + C.** Ship the tool AND validate against OpenAI's audit. This is achievable, honest, and provides real value.
