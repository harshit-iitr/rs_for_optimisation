# AGENT BRIEF — The Stiffness Axis (Paper A, breadth-first campaign)

**Target compute:** 2× NVIDIA A6000, 2–3 wall-clock hours.
**Repo state:** blank slate. Everything below is spec, not description of existing code.
**Prime directive:** *no result is trusted until this repo produces it.* Prior reports exist but are treated as hypotheses, not baselines.

---

## 0. What we are testing and why

The method under study is **Radial Suppression (RS)**: a soft penalty pulling hidden activations toward a √d-radius hypersphere.

```
L_total = L_task(f(x;θ), y) + λ · L_RS(h)
L_RS(h) = (1/d) · (‖h‖₂ − √d)²
```

**The framing that makes this novel.** RS is the Lagrangian relaxation of a hard hyperspherical constraint. λ is a *stiffness* parameter interpolating a continuum:

- λ = 0 → standard ERM
- λ finite → soft retraction toward the sphere
- λ → ∞ → hard projection onto S^{d−1}(√d) (this is what SimbaV2 / nGPT do architecturally)

Hard hyperspherical constraints are published and successful (SimbaV2, ICML 2025 spotlight, in RL; nGPT for transformers). **The soft regime and the stiffness axis itself are unstudied.** That is the gap.

**Theory (T1).** Decomposing the flow ḣ = −∇_h L_total into radial and tangential parts:

```
ḣ_rad = −P_r ∇_h L_CE − (2λ/d)(‖h‖₂ − √d) · h/‖h‖₂
ḣ_tan = −(I − P_r) ∇_h L_CE
```
where P_r = hhᵀ/‖h‖₂². As λ→∞ the radial force becomes infinitely stiff and dynamics reduce exactly to Riemannian gradient flow on the sphere.

**Prediction M1 (the key quantitative test).** Let u = ‖h‖₂ − √d. Linearizing the radial dynamics:
```
u̇ ≈ −(2λ/d)·u + g_rad
```
giving equilibrium radial excess **u\* = g_rad·d / (2λ)** and relaxation timescale **τ_rad = d/(2λ)**.

→ **On a log-log plot of measured steady-state |‖h‖₂ − √d| against λ, the slope should be −1.**

This is the single most valuable measurement in the campaign. It converts T1 from analogy into verified mechanism, it is nearly free to compute, and it is cleanly falsifiable.

**Prediction M2.** The constraint only bites if τ_rad is short relative to the per-task step budget K. Predicts a *critical stiffness* λ\* below which the penalty is inert, scaling with K and learning rate. Expect a knee in the sweep, not a monotone curve.

**Prediction M3.** Radial energy is a *dynamical* quantity (where gradient energy goes); representation rank is a *static* one (what the representation looks like). Recent work (Wang et al. 2026) constructs counterexamples where rank-based diagnostics fail to predict trainability. If Φ̃_rad survives where rank fails, it belongs to a different family than the refuted diagnostics.

---

## 1. CRITICAL: the Φ̃_rad definition bug

Prior internal work reported Φ̃_rad **increasing** ~3× under the penalty (1.09 → 3.17). This is almost certainly because the metric was computed on the *total* gradient, which includes the penalty term — and the penalty gradient is **purely radial by construction**. The metric was measuring itself.

**Correct definition. Φ̃_rad MUST be computed on the task-loss gradient ONLY.**

```python
# Normalized Fractional Radial Energy — CORRECT IMPLEMENTATION
#
# g = ∇_h L_task   <-- TASK LOSS ONLY. NEVER include the RS penalty gradient.
#
# radial component:  g_rad = (g · ĥ) ĥ   where ĥ = h / ‖h‖₂
# Φ_rad  = ‖g_rad‖² / ‖g‖²
# Φ̃_rad = d · Φ_rad          (normalized: random gradient in R^d gives E[Φ_rad] = 1/d, so E[Φ̃_rad] = 1)
#
# Φ̃_rad = 1  → radial energy at chance level
# Φ̃_rad > 1  → gradient energy concentrated in the radial direction
```

Implementation note: obtain ∇_h L_task via a **separate backward pass on the task loss alone**, or by registering a hook that captures the task-loss gradient before the penalty gradient is accumulated. Do not subtract analytically — compute it cleanly and assert independence.

**Mandatory unit test** (`tests/test_phi_rad.py`):
1. Random Gaussian g and random h in R^d → Φ̃_rad ≈ 1.0 (tolerance 0.15, averaged over 1000 draws).
2. g set exactly parallel to h → Φ̃_rad == d.
3. g set exactly orthogonal to h → Φ̃_rad == 0.
4. **With λ > 0 active, assert the logged Φ̃_rad is numerically identical to the λ=0 computation on the same (h, g_task) pair.** If this fails, the penalty gradient is leaking into the metric. Halt and fix.

Report Φ̃_rad **per layer** and as a network mean.

---

## 2. Repo layout

```
stiffness/
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── s1_lambda_sweep.yaml
│   ├── s2_baselines.yaml
│   ├── s4_scaling.yaml
│   ├── s5_transfer.yaml
│   └── s6_limits.yaml
├── src/
│   ├── data/
│   │   ├── permuted_mnist.py
│   │   ├── rotating_mnist.py
│   │   └── split_cifar100.py
│   ├── models/
│   │   ├── mlp.py
│   │   └── convnet.py
│   ├── methods/
│   │   ├── rs.py              # the soft penalty + hard-projection variant
│   │   ├── baselines.py       # L2, LN, LN+L2, S&P, L2-init, ER penalty
│   │   ├── cbp.py             # continual backprop
│   │   └── redo.py            # dormant-neuron recycling
│   ├── metrics/
│   │   ├── phi_rad.py         # SEE SECTION 1 — most important file in repo
│   │   ├── rank.py            # effective rank, stable rank, numerical rank
│   │   ├── neurons.py         # dead + dormant fractions
│   │   ├── norms.py           # activation radius, weight norm, per-layer
│   │   └── readiness.py       # optimization-readiness proxy (see 4.3)
│   ├── train.py               # single-run entry point
│   └── sweep.py               # parallel launcher, 2-GPU aware
├── tests/
│   ├── test_phi_rad.py        # MANDATORY, see section 1
│   ├── test_rs_gradient.py    # penalty gradient is nonzero and enters the graph
│   └── test_rank.py
├── analysis/
│   ├── s1_stiffness_curve.py
│   ├── s3_diagnostic_race.py
│   └── s4_lambda_scaling.py
└── results/
    └── <run_id>/metrics.parquet
```

---

## 3. Common setup

### 3.1 Primary benchmark — Permuted MNIST

- 3-layer MLP, widths **1000/1000/1000**, Kaiming uniform init, ReLU by default.
- Each task = fresh pixel permutation. Labels unchanged.
- **150 tasks**, 1 epoch per task, batch 256 (≈235 steps/task, K ≈ 235).
- Optimizer: **SGD, lr = 1e-2, no momentum, no LR annealing** for the main grid. (SGD keeps the flow analysis in section 0 honest; AdamW's preconditioner breaks the correspondence. Run AdamW only in S6.)
- Grad clip 0.5.
- Single shared output head across tasks (no per-task head reset — resetting confounds plasticity measurement).
- Load full MNIST onto GPU once; permutations are index gathers. Data loading must not be the bottleneck.

### 3.2 Where RS is applied

Default: **pre-activations of all hidden layers**. d = layer width. Log a `rs_layers` field so scope is always recoverable.

Hard-projection variant (`λ = inf`): after computing h, replace with `h ← √d · h/‖h‖₂`, straight-through in the backward. This is the SimbaV2-style limit and is a *required* arm of S1.

### 3.3 Metrics — logged once per task boundary, on a held-out probe batch of 2000 samples

| Metric | Definition | Notes |
|---|---|---|
| `phi_rad_tilde[l]` | §1 | task-gradient only; per layer + mean |
| `radial_excess[l]` | mean(‖h‖₂) − √d | signed; the M1 quantity |
| `radius_mean[l]`, `radius_std[l]` | mean and std of ‖h‖₂ | |
| `eff_rank[l]` | exp(−Σ pᵢ log pᵢ), pᵢ = σᵢ/Σσⱼ | on B×d activation matrix, B=2000 |
| `stable_rank[l]` | ‖A‖²_F / ‖A‖²₂ | |
| `dead_frac[l]` | fraction of units with zero output on all probe inputs | |
| `dormant_frac[l]` | fraction with normalized activation score < 0.025 | Sokar et al. definition |
| `weight_norm[l]` | ‖W‖_F | |
| `grad_norm[l]` | ‖∇_W L_task‖_F | |
| `readiness` | §4.3 | |
| `train_acc`, `test_acc` | end of task | |
| `first_epoch_gain` | acc after 1 epoch on task t − acc at task-t init | **the trainability target** |

Write one row per (run_id, task, layer) to parquet. Every run logs a full config hash.

### 3.4 Seeds and statistics

- **Minimum 3 seeds** everywhere; 5 for headline comparisons.
- Every reported comparison: paired by seed, with **explicit test statistic, p-value, n, and sign split**.
- `analysis/` must emit p-values automatically. A comparison without a test does not go in a table.

---

## 4. Experiment suites

### S0 — Infrastructure and sanity (target 20 min)

| Run | Spec | Pass criterion |
|---|---|---|
| S0.1 | Unit tests | All green, esp. `test_phi_rad.py` item 4 |
| S0.2 | λ=0 baseline, 3 seeds, 150 tasks | Test acc degrades and dead_frac rises over tasks — i.e. plasticity loss is *reproduced* before anything is claimed about fixing it |
| S0.3 | λ=0.05, 1 seed, 20 tasks | RS loss term is **nonzero and changing** every step (log it explicitly); radius_mean moves toward √d |

**Halt condition:** if S0.2 shows no plasticity loss, the benchmark setup is wrong and nothing downstream is interpretable.

---

### S1 — The stiffness sweep ★ CORE

**λ ∈ {0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, inf}** — 13 values × 3 seeds = **39 runs**.

Permuted MNIST, 150 tasks, full metric logging.

Outputs:
- **Stiffness curve**: mean final-20-task accuracy vs λ (log x-axis), with hard-projection arm marked.
- Same curve for dead_frac, eff_rank, phi_rad_tilde.

**Questions this answers:**
1. Is the optimum **interior** (finite λ beats both ERM and hard projection)? If yes, that is the paper's headline and it is a claim nobody has made.
2. Is there a **knee** at critical λ\* (M2)?
3. Does hard projection underperform soft? (Would be a genuinely novel negative for the SimbaV2 line.)

---

### S2 — Baselines head-to-head ★ CORE

Same benchmark, 150 tasks, **5 seeds each**:

| Arm | Config |
|---|---|
| BP | plain SGD |
| L2 | weight decay ∈ {1e-3, 1e-4}, pick best on IID val |
| LayerNorm | LN before each nonlinearity |
| **LN + L2** | the field's strongest general baseline |
| S&P | shrink 0.8, noise σ=0.01 at task boundary |
| L2-init | regularize toward θ₀, α=0.01 |
| CBP | replacement rate 1e-6, maturity 100 |
| ReDo | dormant threshold 0.025, recycle every 1000 steps |
| ER penalty | explicit effective-rank penalty (He et al.'s ER arm) |
| L2 + ER | He et al.'s L2-ER — **the strongest published competitor** |
| **RS** | best λ from S1 |
| RS + CBP | combination |

~12 arms × 5 seeds = **60 runs**.

**Context the agent should know:** published work reports that on *Permuted MNIST specifically*, LN+L2 underperforms relative to other benchmarks, and the pure ER penalty loses plasticity across all environments. Both are favourable priors for RS — but they are priors, not results. Measure them here.

---

### S3 — Diagnostic horse race ★ CORE (analysis only, no new training)

Using checkpoints already logged in S1 + S2:

For every (run, task t) pair, correlate each diagnostic measured at end-of-task-t against **`first_epoch_gain` on task t+1**:

- `phi_rad_tilde` (M3's candidate)
- `eff_rank`, `stable_rank`
- `dead_frac`, `dormant_frac`
- `weight_norm`
- `readiness`

Report **Spearman and Kendall** (not just Pearson — the relationship need not be linear), with n and p, computed:
1. within the λ=0 arm alone,
2. pooled across all arms,
3. **held out across benchmarks** (fit on Permuted MNIST, test on S5's second benchmark).

**This is the test of A1.** If Φ̃_rad ranks trainability better than rank-based diagnostics, M3 is supported and the paper has a diagnostic contribution that survives the published counterexamples against rank.

#### 4.3 Optimization-readiness proxy

Wang et al. define readiness as gradient strength × gradient reliability. Implement a tractable proxy on the probe batch: split into m=8 microbatches, compute per-microbatch task gradients gᵢ, then

```
strength    = ‖mean(gᵢ)‖₂
reliability = ‖mean(gᵢ)‖₂² / mean(‖gᵢ‖₂²)     # ∈ (0,1], gradient agreement
readiness   = strength · reliability
```
Document this as *our* proxy, not their exact metric.

---

### S4 — The 1/λ scaling law ★ HIGH VALUE, near-free

Reuses S1 runs plus a small dedicated grid.

**S4.1 — M1 verification.** Plot `|radial_excess|` at end-of-task against λ, log-log, per layer. **Predicted slope −1.** Fit and report slope with CI. Do this at tasks {10, 50, 100, 150} to check stability over training.

**S4.2 — timescale.** From a mid-training checkpoint, freeze the data, perturb activations radially by +20%, and measure the decay of u under continued training. Fit exponential; predicted τ_rad = d/(2λ). Run at λ ∈ {1e-2, 0.1, 1.0}, 3 seeds. 9 short runs.

**S4.3 — width scaling.** Does the √d target hold as d changes? Widths ∈ {256, 512, 1000, 2048}, λ at S1 optimum, 3 seeds. 12 runs. Tests whether λ must be retuned per width — a practical question no one has answered, and a reviewer will ask.

---

### S5 — Second benchmark (generalization of the claims)

**S5.1 — Rotating MNIST.** 2-layer MLP width 256, continuous rotation 0°→180° over 100 epochs. λ ∈ {0, best, 10×best}, 3 seeds. 9 runs. Re-run the S3 correlation here — **A1 needs replication on a second benchmark or it is not a claim.**

**S5.2 — Split CIFAR-100.** 4-layer ConvNet, no BatchNorm, 10 tasks × 10 classes. Arms: BP, LN+L2, CBP, RS(best λ). 3 seeds. 12 runs. Establishes the result isn't MLP-only.

**S5.3 — ReLU vs Leaky ReLU.** Prior internal work claims standard ReLU makes RS unstable (collapse to 10% accuracy, 100% dead units) and Leaky ReLU (α=0.01) fixes it. **Verify or refute.** Permuted MNIST, {ReLU, LeakyReLU(0.01)} × {λ=0, λ=best}, 3 seeds. 12 runs. If real, it's a required-conditions result; if not, drop the claim.

---

### S6 — Limits and failure modes

**S6.1 — LR robustness.** lr ∈ {1e-3, 1e-2, 1e-1}, arms {BP, RS, L2-init}, 3 seeds. 27 runs. Expect RS to provide no weight-space stability at high lr.

**S6.2 — Optimizer interaction.** AdamW vs SGD at λ=best. Does the flow analysis survive preconditioning? 6 runs.

**S6.3 — Stability tax.** Log accuracy on **all previous tasks** at every task boundary for {BP, RS, CBP, L2-init}, 3 seeds. Produces the plasticity/stability frontier. This goes in the paper as an honest limitation, not an appendix.

---

## 5. Compute budget

MLP runs are launch-overhead-bound, not FLOP-bound. Estimate ~90–120 s per 150-task Permuted MNIST run including metric logging (the per-task SVD on a 2000×1000 matrix is ~50 ms; 150 of them is negligible).

| Suite | Runs | Est. GPU-min |
|---|---|---|
| S0 | 5 | 8 |
| S1 | 39 | 70 |
| S2 | 60 | 110 |
| S3 | 0 (analysis) | 5 |
| S4 | 21 | 35 |
| S5 | 33 | 90 (CIFAR heavier) |
| S6 | 33 | 60 |
| **Total** | **191** | **~380 GPU-min** |

2 GPUs × 180 min = 360 GPU-min, and MLPs allow 4–6 concurrent runs per A6000. **Comfortable inside the budget with headroom for reruns.**

**Execution order:** S0 → S1 → S4.1 → S2 → S3 → S5 → S6. S4.1 is early and out of order because it is cheap and it validates the theory; if M1's slope is not ≈ −1, we want to know before spending on S2.

---

## 6. Kill criteria — report these immediately, do not work around them

| Suite | Kill condition | Consequence |
|---|---|---|
| S0.2 | No plasticity loss in baseline | Benchmark broken. Halt everything. |
| S0.3 | RS loss term is zero / constant | Penalty not in graph. Halt. |
| S1 | Stiffness curve flat across 5 orders of magnitude | No stiffness story; pivot to M1 + S3 as the paper |
| S4.1 | Fitted slope not in [−1.3, −0.7] | T1's radial dynamics do not describe the system. **Report loudly** — this is a legitimate negative result and it reshapes the paper. |
| S2 | RS not distinguishable from LN+L2 (paired p > 0.1) | No method contribution; paper becomes diagnostics-only (S3 + S4) |
| S3 | Φ̃_rad ranks trainability no better than eff_rank | A1 dies; M3 refuted |
| S5.1 | S3 correlation fails to replicate | A1 dies. Non-negotiable. |

**A failed suite is a result.** Log it, don't retune around it. Four of these can fail independently and there is still a paper; what kills the project is discovering a failure in week two that was visible in hour two.

---

## 7. Rules for the agent

1. **Never report a comparison without a significance test.** Paired where seeds align, with n and sign split stated.
2. **Never tune on the target metric.** λ selection happens once, on IID validation, in S1. It is then frozen for S2, S4.3, S5, S6.
3. **Never freeze layers upstream of a penalized activation.** ∂L_RS/∂w = 0 for any parameter h doesn't depend on — such a run measures nothing. Assert against it at construction time.
4. **Log the RS penalty value every step** in at least one run per suite. Silent inertness has already burned this project once.
5. **Identical values to 4 decimals across arms are a bug signal, not a finding.** Flag and investigate.
6. **Every plot gets error bars.** Every table gets ±std and n.
7. Set seeds for torch, numpy, and python random. Log versions, git SHA, and full config with every run.
8. When something surprises you, write it in `results/ANOMALIES.md` rather than silently smoothing it.

---

## 8. Key references for context

- Dohare et al., *Loss of plasticity in deep continual learning*, Nature 632:768–774, 2024. (CBP; the problem statement)
- Lyle et al., *Disentangling the Causes of Plasticity Loss in Neural Networks*, arXiv:2402.18762, 2024. (LN+L2 as strongest general baseline; parameter norm growth)
- Prakash, He, Guo et al., *Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning*, ICML 2026, arXiv:2509.22335. (L2-ER; ε-rank; **closest competitor** — note their ER-alone arm fails, and LN+L2 underperforms on Permuted MNIST)
- Wang et al., *Predicting Plasticity in Deep Continual Learning: A Theoretical Perspective*, arXiv:2605.09044, 2026. (counterexamples against rank-based diagnostics; optimization readiness — **the threat to A1**)
- Lee et al., *Hyperspherical Normalization for Scalable Deep Reinforcement Learning* (SimbaV2), ICML 2025 spotlight, arXiv:2502.15280. (**hard** hyperspherical constraint; the λ→∞ limit of our method)
- Loshchilov et al., *nGPT: Normalized Transformer with Representation Learning on the Hypersphere*, arXiv:2410.01131, 2024.
- Sokar et al., *The Dormant Neuron Phenomenon in Deep RL*, ICML 2023. (ReDo; dormant-unit definition)
- Abbas et al., *Loss of plasticity in continual deep RL*, CoLLAs 2023. (CReLU)
- Ash & Adams, *On warm-starting neural network training*, NeurIPS 2020. (Shrink & Perturb)