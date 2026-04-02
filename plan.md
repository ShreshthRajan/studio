# FINAL PLAN: Environment Selection as Experimental Design
## An Information-Theoretic Theory of RL Learnability

---

## GROUND TRUTH (What We Now Know For Certain)

### Confirmed ✓
1. **The gradient IS** ∇J = p(1-p) · (g⁺ - g⁻), where (g⁺ - g⁻) is the "Gradient Gap" vector (Suk & Duan, 2025)
2. **p(1-p) = 1/I_B(p)** where I_B is Fisher information of Bernoulli — this is a mathematical identity
3. **GRPO normalization preserves the p(1-p) structure** — discordant pairs occur with probability 2p(1-p) (U-Statistic paper)
4. **The OED-RL duality is NOVEL** — 30+ targeted searches confirm nobody has made this connection (OED novelty agent)
5. **VIP (ICLR 2026) already uses p(1-p) and Neyman allocation** — the functional form is known, but NOT framed as OED or Fisher info
6. **VCRL (2025) uses reward variance as curriculum heuristic** — no theoretical grounding via OED
7. **Mukherjee et al. (NeurIPS 2024) applies Kiefer-Wolfowitz to RLHF** — different design variable (preference queries, not training tasks)
8. **R2E-Gym works, SWE-Smith doesn't** — SWE-Smith failed due to "high solve-none rate" (p ≈ 0 for weak models)
9. **SWE-Smith difficulty buckets**: Easy 58.6%, Medium 41.0%, Hard 17.0% (for Claude 3.7 — much lower for weaker models)

### Threats ✗
1. **"Imbalanced Gradients" paper (Meta, arXiv:2510.19178)**: Gradient magnitude does NOT correlate with learning gains ACROSS tasks. Large-gradient tasks can have lower learning gains than small-gradient ones. **This means p(1-p) may not predict cross-task/cross-environment training value quantitatively.**
2. **The Gradient Gap (g⁺ - g⁻) is high-dimensional and task-dependent.** Our scalar p(1-p) theory ignores the directional component. The Gradient Gap itself varies across tasks in ways we don't model.
3. **Edge of Competence theory doesn't transfer.** It's specific to group composition on finite groups via Fourier analysis. We CANNOT build dynamics theorems on their formalism.
4. **No per-task solve rate data is published** for any code environment. We can't easily validate empirically.
5. **All SWE environments require Docker.** MacBook compute is very limited for rollouts.

### What This Means For Our Paper
- The OED DUALITY (Contribution 1) is **safe and novel** — it's a mathematical observation that doesn't depend on cross-task prediction
- The OPTIMAL DISTRIBUTION (Contribution 2) is **safe** — it's a variational result
- The LEARNABILITY SCORE as a quantitative predictor is **questionable** — Imbalanced Gradients suggests magnitude ≠ learning gains across tasks
- The DYNAMICS (Part III) should be **dropped** — formalism doesn't transfer
- **Reframe**: ELS is a NECESSARY condition (ELS ≈ 0 → no learning, guaranteed), not a SUFFICIENT condition (high ELS → good learning, not guaranteed). This is still very useful — it catches the SWE-Smith failure case cheaply.

---

## THE 5-STEP PLAN

### STEP 1: Read Foundation Papers + Prove Core Duality (Week 1)
**What we do in the next prompt session.**

```
Day 1-2: Read the exact math
  - Read Gradient Gap paper (arXiv:2510.08539) — extract exact Theorem 1 expression
  - Read GRPO U-Statistic paper (arXiv:2603.01162) — extract exact MSE bounds
  - Read VIP paper (arXiv:2602.01601) — confirm if they mention Fisher info / OED
  - Read Imbalanced Gradients paper (arXiv:2510.19178) — understand the threat precisely
  - Read Mukherjee et al. (arXiv:2404.13895) — understand their KW generalization

Day 3-5: Prove Theorem 1 (The Duality)
  - Start from ∇J = p(1-p) · (g⁺ - g⁻)
  - Observe p(1-p) = 1/I_B(p) (Fisher information identity)
  - State the duality: RL-optimal task selection maximizes E_D[1/I_B(p)],
    OED maximizes E_D[I_B(p)] — these are dual objectives
  - Check: does the Kiefer-Wolfowitz equivalence theorem have an analog
    for the reciprocal Fisher information? Work through the math.

Day 5-7: Prove Theorem 2 (Optimal Difficulty Distribution)
  - Set up variational problem: max_D ELS(D, K) where
    ELS = G(D)² / V(D, K)
  - G(D) = E_D[p(1-p)] (environment gradient signal)
  - V(D, K) = E_D[σ²(p,K)] + Var_D[p(1-p)·||g⁺-g⁻||]
    (within-task + between-task variance)
  - Solve Euler-Lagrange for D*(p)
  - Characterize D* as function of K (GRPO group size)
  - Derive limiting cases: K→∞, K=2
```

**GO/NO-GO AFTER STEP 1:**

| Outcome | What Happened | Rating | Action |
|---------|--------------|--------|--------|
| **BANGER** | KW analog yields non-trivial equivalence. D*(p) has elegant closed form that depends on K in a surprising way. The duality produces at least one non-obvious corollary. | **8.5/10** | Proceed full speed |
| **STRONG** | Duality is clean but KW analog is trivial. D*(p) is just "concentrate at 1/2" regardless of K. | **7/10** | Proceed, focus on the duality framing + IRT connection |
| **WEAK** | The between-task variance term in V(D,K) involves ||g⁺-g⁻|| which we can't characterize without assumptions, making ELS uncomputable from the theory alone. | **5/10** | Simplify: drop ELS, focus purely on the OED duality as a conceptual contribution |
| **DEAD** | VIP paper already explicitly names Fisher information and OED. Our novelty claim collapses. | **3/10** | Kill this specific angle. Pivot entirely. |

**What "interesting results" look like after Step 1:**
- A clean theorem statement: "The RL-optimal task distribution is the anti-D-optimal design"
- A plot showing D*(p) for K=2, 4, 8, 16, 64 — showing how the optimal distribution narrows as K increases
- A clear statement of the duality that fits in one tweet

**Local feasibility**: Step 1 is pure math. Requires: LaTeX, paper reading, pencil. No compute, no Docker, no GPUs.

---

### STEP 2: Develop the IRT Connection + Address Imbalanced Gradients (Week 2-3)

```
The IRT (Item Response Theory) connection is potentially richer than generic OED:

IRT Mapping:
  - "Examinee ability" θ  ↔  RL "policy quality" π
  - "Test item"           ↔  RL "training task"
  - "Correct response"    ↔  RL "task solved"
  - "Adaptive testing"    ↔  RL "curriculum design"
  - "Optimal test assembly" ↔ RL "environment selection"

IRT's Fisher information for 2PL model:
  I(θ) = a² · P(θ)(1-P(θ))
  where a = discrimination parameter, P(θ) = success probability

This is EXACTLY our setting. 50+ years of IRT theory gives us:
  - Computerized Adaptive Testing algorithms → online curriculum design
  - Optimal test assembly theory → environment selection
  - Item information functions → task learnability functions
  - Test information = sum of item informations → environment = sum of task signals

Nobody has made this IRT → RL mapping (confirmed by search).

Address the Imbalanced Gradients threat:
  - The Gradient Gap (g⁺ - g⁻) plays the role of the IRT "discrimination parameter" a
  - High-discrimination items (large ||g⁺ - g⁻||) provide more information
  - The Imbalanced Gradients finding = "discrimination varies across tasks"
  - Our theory becomes: ELS ∝ E_D[a² · p(1-p)] not just E_D[p(1-p)]
  - To estimate a, we need gradient information — but we can BOUND it:
    if a > 0 for all tasks, then E_D[a² · p(1-p)] > a_min² · E_D[p(1-p)]
  - So ELS (ignoring discrimination) is a LOWER BOUND on the true signal
  - This makes ELS a conservative estimate — it can't overpredict
```

**Deliverable**: Extended theory incorporating the discrimination parameter from IRT, addressing the Imbalanced Gradients concern, and importing specific IRT results (optimal test assembly algorithms, adaptive testing convergence rates).

---

### STEP 3: Empirical Validation with Available Data (Week 3-4)

```
What we CAN do without a GPU cluster:

3a. Theoretical ELS from published data:
  - SWE-Smith: published 3-bucket difficulty (58.6/41.0/17.0% for Claude 3.7)
  - For a Qwen-2.5-1.5B model, estimate these would be ~5/2/0.5% (rough scaling)
  - Compute G(D_SWE-Smith) = E[p(1-p)] from these estimates
  - Compare to R2E-Gym: assume smoother distribution (R2E-Gym was designed for it)
  - Show: G(D_R2E-Gym) >> G(D_SWE-Smith) ≈ 0 for weak models

3b. Synthetic validation:
  - Generate synthetic task distributions with known properties
  - Simulate GRPO training dynamics on these distributions (bandit setting)
  - Show: ELS predicts which synthetic environments train faster
  - This requires NO real SWE environments — just simulated Bernoulli bandits
  - Can run on a laptop in minutes

3c. Compare D* predictions to empirical "what works":
  - DeepSWE trained on R2E-Gym subset (4,500 tasks)
  - If R2E-Gym's difficulty distribution approximates D*, that explains its success
  - If SWE-Smith's distribution is far from D*, that explains its failure

3d. Plot the duality visually:
  - D-optimal design vs RL-optimal design on [0,1]
  - Optimal D*(p) for different K values
  - Fisher information I(p) vs gradient signal 1/I(p)
  - These are compelling figures for the paper
```

**Local feasibility**: 3a and 3d are pure computation (Python on laptop). 3b requires simulating GRPO on synthetic bandits (Python on laptop, no LLM needed). 3c requires published data analysis.

---

### STEP 4: Write the Paper (Week 4-6)

```
Title: "Environment Selection as Experimental Design:
        An Information-Theoretic Theory of RL Task Learnability"

Structure:
1. Introduction
   - The R2E-Gym vs SWE-Smith puzzle
   - Our answer: environment selection is dual to experimental design
   - The Learnability Paradox: you learn most from tasks you can least predict

2. Preliminaries
   - GRPO/REINFORCE with binary rewards
   - Gradient Gap decomposition (cite Suk & Duan)
   - Basics of optimal experimental design (Fisher info, D-optimality)

3. The Gradient-Information Duality
   - Theorem 1: ∇J = p(1-p) · ΔG, and p(1-p) = 1/I_B(p)
   - Theorem 2: RL-optimal ≠ D-optimal (dual objectives)
   - Connection to IRT: discrimination parameter = ||ΔG||
   - Corollary: KW equivalence analog (or show why it doesn't hold —
     either way is interesting)

4. Optimal Task Distributions
   - Theorem 3: D*(p) via calculus of variations
   - Characterization as function of K
   - Comparison to Neyman allocation (VIP) — our result is more general
   - Compute-optimal allocation across environments (Theorem 4)

5. Environment Learnability Score
   - Definition: ELS as signal-to-noise ratio
   - Properties: ELS = 0 iff all tasks trivial or impossible
   - Interpretation: ELS is a necessary condition for learning
   - Discussion of limitations (Imbalanced Gradients — ELS ignores discrimination)

6. Empirical Analysis
   - Synthetic bandit experiments validating the theory
   - ELS computation for R2E-Gym vs SWE-Smith from published data
   - D* visualization for different K values

7. Discussion
   - Connection to IRT and Computerized Adaptive Testing
   - Importing 50 years of test theory into RL
   - Limitations and future work

Target: NeurIPS 2026 (deadline ~May), ICML 2026, or standalone arXiv preprint
```

---

### STEP 5: Tool + Pitch (Week 7-8)

```
Open-source tool: envscale
  - Computes ELS from difficulty distribution data
  - Computes optimal D*(p) for given K
  - Computes cross-environment allocation
  - Includes synthetic bandit simulator for validation

Pitch to OpenAI friend:
  - "I proved that RL environment selection is dual to classical
     experimental design — a field with 50 years of theory you haven't
     been using."
  - "Here's the formula. Here's why R2E-Gym works and SWE-Smith doesn't."
  - "Give me your per-task solve rate data and I'll compute the optimal
     difficulty distribution for your GRPO group size."
  - "I want to come prove the next set of theorems on your data."
```

---

## WHAT STEP 1 LOOKS LIKE IN THE NEXT PROMPT

We will:
1. **Fetch and read** the exact math from the Gradient Gap paper, GRPO U-Statistic paper, and VIP paper
2. **Write the proof** of Theorem 1 (the duality) in LaTeX
3. **Attempt the variational problem** for D*(p) (Theorem 3)
4. **Produce the first visualization**: D-optimal vs RL-optimal, D*(p) for different K

This is **math + LaTeX + simple Python plots**. No Docker, no GPUs, no SWE environments.

**What "interesting results" means after Step 1:**
- A proven theorem that nobody has stated before
- A formula for D*(p|K) that VIP doesn't have
- A plot showing the duality visually
- A clear 1-paragraph explanation of the Learnability Paradox

---

## PAPERS TO READ (ordered by priority for Step 1)

| Priority | Paper | Why | What to extract |
|----------|-------|-----|----------------|
| **P0** | [Gradient Gap (2510.08539)](https://arxiv.org/abs/2510.08539) | The foundational decomposition | Exact Thm 1 expression, step-size formula |
| **P0** | [GRPO U-Statistic (2603.01162)](https://arxiv.org/abs/2603.01162) | MSE bounds we need for V(D,K) | Exact variance formula as function of K |
| **P0** | [VIP (2602.01601)](https://arxiv.org/abs/2602.01601) | Closest competitor — must confirm they DON'T name Fisher/OED | Exact optimization formulation |
| **P1** | [Imbalanced Gradients (2510.19178)](https://arxiv.org/abs/2510.19178) | The main threat to address | When does magnitude ≠ learning? Why? |
| **P1** | [Mukherjee et al. (2404.13895)](https://arxiv.org/abs/2404.13895) | KW for RLHF — must differentiate | Their KW generalization |
| **P1** | [VCRL (2509.19803)](https://arxiv.org/abs/2509.19803) | Uses p(1-p) as curriculum heuristic | How they use variance, any theory? |
| **P2** | [Online Difficulty Filtering (2504.03380)](https://arxiv.org/abs/2504.03380) | Proves improvement lower-bounded by variance | Their exact bound — may overlap with our Thm |
| **P2** | [85% Rule (Nature 2019)](https://www.nature.com/articles/s41467-019-12552-4) | Optimal error rate for learning | Their derivation — different from ours? |
| **P2** | [Edge of Competence (2602.14872)](https://arxiv.org/abs/2602.14872) | Relay effect concepts | Conceptual only — formal machinery doesn't help |
| **P3** | Pukelsheim (2006) Optimal Design of Experiments | OED theory toolkit | D-optimal, KW theorem, equivalence results |
| **P3** | Lord (1980) Applications of IRT | IRT toolkit | Optimal test assembly, information functions |

---

## HONEST FINAL RATING: 7.5/10

**Why 7.5:**
- The OED duality is **confirmed novel** and genuinely interesting
- The optimal distribution D*(p|K) is a **real mathematical contribution** (variational result VIP doesn't have)
- The IRT connection opens a **rich vein** of importable results
- The Learnability Paradox framing is **memorable and tweetable**
- It **explains a real empirical puzzle** (R2E-Gym vs SWE-Smith)
- It's **pure math** — doable on a laptop, plays to your strengths
- The paper **positions you perfectly** for an RL theory role at a lab

**Why not 8+:**
- **Imbalanced Gradients is a real threat**: p(1-p) magnitude ≠ cross-task learning gains. Our theory is a NECESSARY condition, not sufficient. This limits the practical impact.
- **VIP already uses p(1-p) + Neyman allocation**: The basic functional form is known in the RL community. We're adding the OED/Fisher framing, not discovering the relationship.
- **Empirical validation is thin**: We can't run real SWE environments. Synthetic bandits are OK but not compelling for reviewers who want real-world results.
- **D*(p) might just be "concentrate at 1/2"**: If the optimal distribution is obvious/boring regardless of K, the variational result is uninteresting.
- **The duality can be stated in 3 lines**: The core observation is simple. The question is whether we can build enough depth around it.

**What pushes to 8.5+:**
- D*(p|K) is non-trivial and K-dependent in a surprising way
- The KW analog yields unexpected corollaries
- The IRT connection produces specific, actionable algorithms (imported from CAT)
- A lab validates ELS on internal data
- The "Online Difficulty Filtering" paper's bound is strictly weaker than ours

**What would kill it:**
- VIP paper explicitly mentions Fisher information and OED
- D*(p) = δ(p-1/2) for all K (trivial result)
- Reviewers say "this is just p(1-p), everyone knows this"
