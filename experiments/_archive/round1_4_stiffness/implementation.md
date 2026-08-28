# Stiffness Axis (Radial Suppression) - Implementation Plan

This document details the step-by-step implementation plan for the Stiffness Axis repository, designed to meet the rigorous requirements and kill criteria outlined in `plan.md`. The implementation is broken down into three carefully structured phases to ensure logic, metrics, and methodology are foolproof.

## Guiding Principles
- **No result is trusted until reproduced locally.** 
- **Metric purity is paramount**, especially for `phi_rad_tilde`. Penalty gradients must never leak into the task-loss gradient.
- **Failures are results.** We will rigorously evaluate kill criteria at every phase and halt if violated, rather than silently tuning around them.
- **Statistical rigor.** All comparisons will have paired significance tests, n, p-values, and sign splits.

---

## Phase 1: Foundation, Metric Purity, and Sanity Checks (S0)

**Goal:** Establish the repository, build the baseline data/model pipeline, ensure `phi_rad_tilde` is implemented perfectly, and reproduce baseline plasticity loss (kill criterion check).

### 1.1 Project Structure & Environment
- Create the repository layout as defined in `plan.md` (`src/`, `tests/`, `configs/`, `analysis/`, `results/`).
- Create `requirements.txt` with frozen dependencies (PyTorch, NumPy, Pandas, PyArrow, SciPy, Matplotlib) to ensure reproducible data science pipelines.
- Establish baseline logging utilizing `pyarrow` / `pandas` to write to `metrics.parquet`.

### 1.2 Data and Architecture
- Implement `src/data/permuted_mnist.py`: Load full MNIST onto GPU memory once. Permutations are implemented as index gathers for speed. Batch size 256.
- Implement `src/models/mlp.py`: 3-layer MLP (1000/1000/1000), Kaiming uniform initialization, standard ReLU, and a **single shared output head**.

### 1.3 Core Method and Metrics Implementation (CRITICAL)
- **`src/methods/rs.py`**: Implement the soft Radial Suppression (RS) penalty and the hard hyperspherical projection variant (using a `StraightThrough` custom autograd function for `λ=inf`).
- **`src/metrics/phi_rad.py`**: Implement `phi_rad_tilde`. To guarantee the RS penalty gradient does not leak into the measurement, we will use PyTorch's `register_full_backward_hook` or a dedicated `torch.autograd.grad` pass explicitly on `L_task` to compute `∇_h L_task` cleanly.
- Implement secondary metrics in `src/metrics/`:
  - `rank.py`: Effective rank and stable rank on the probe batch.
  - `neurons.py`: Dead and dormant fractions.
  - `norms.py`: Activation radius (`mean(‖h‖₂)`), weight norm, gradient norm.
  - `readiness.py`: Gradient strength and reliability proxy across microbatches.

### 1.4 Unit Tests
- Implement `tests/test_phi_rad.py` as mandated:
  - Test 1: Random Gaussian $g$, random $h$ -> $\tilde{\Phi}_{rad} \approx 1.0$.
  - Test 2: $g \parallel h$ -> $\tilde{\Phi}_{rad} = d$.
  - Test 3: $g \perp h$ -> $\tilde{\Phi}_{rad} = 0$.
  - Test 4: (Leakage Check) Assert that with $\lambda > 0$, the logged $\tilde{\Phi}_{rad}$ matches exactly the $\lambda=0$ computation on the same $(h, g_{task})$.
- Implement `tests/test_rs_gradient.py` to ensure the penalty gradient correctly flows to weights.

### 1.5 Execution: Sanity Checks (S0)
- **S0.1**: Pass all unit tests.
- **S0.2**: Run $\lambda=0$ baseline (3 seeds, 150 tasks). 
  - *Kill Criterion*: Ensure plasticity loss is reproduced (test accuracy degrades, dead fraction rises).
- **S0.3**: Run $\lambda=0.05$ (1 seed, 20 tasks).
  - *Kill Criterion*: Verify RS loss term is nonzero, changes every step, and radius moves toward $\sqrt{d}$.

---

## Phase 2: Core Experiments & Theory Verification (S1, S4.1, S2, S3)

**Goal:** Execute the primary stiffness sweep, verify the theoretical prediction (M1), and run the baseline head-to-head comparisons.

### 2.1 Training Infrastructure
- **`src/train.py`**: Ensure strict adherence to SGD, `lr=1e-2`, no momentum, no LR annealing, and grad clipping of 0.5. Add probe batch evaluation on 2000 samples at task boundaries.
- **`src/sweep.py`**: Implement a parallel launcher to schedule runs efficiently across the 2 available NVIDIA A6000 GPUs.

### 2.2 S1: The Stiffness Sweep
- Run the 13 defined $\lambda$ values $\times$ 3 seeds on Permuted MNIST (150 tasks).
- Identify the optimal interior $\lambda$.
- *Analysis*: Plot the stiffness curve (Accuracy, dead fraction, rank, $\tilde{\Phi}_{rad}$ vs. $\lambda$).

### 2.3 S4.1: M1 Verification (Theory Check)
- Extract data from S1 to plot steady-state radial excess vs $\lambda$ on a log-log scale.
- Calculate the fitted slope across layers at tasks 10, 50, 100, 150.
- *Kill Criterion*: If the fitted slope is not in $[-1.3, -0.7]$, halt. Theory T1 does not describe the system.

### 2.4 S2: Baselines Head-to-Head
- Implement remaining baselines in `src/methods/baselines.py`, `cbp.py`, and `redo.py` (L2, LN, LN+L2, Shrink & Perturb, L2-init, CBP, ReDo, ER penalty, L2+ER).
- Run 12 arms $\times$ 5 seeds for 150 tasks.
- *Kill Criterion*: If RS is not statistically distinguishable from LN+L2 (paired $p > 0.1$), we flag the method contribution as dead.

### 2.5 S3: Diagnostic Horse Race
- Extract metrics from S1/S2 checkpoints.
- Calculate Spearman and Kendall correlations between end-of-task-$t$ diagnostics and `first_epoch_gain` on task $t+1$.
- *Kill Criterion*: If $\tilde{\Phi}_{rad}$ ranks trainability no better than effective rank, hypothesis A1 dies.

---

## Phase 3: Generalization, Limits, and Scaling (S4.2-3, S5, S6)

**Goal:** Validate scaling laws, ensure the method generalizes to other domains, and document boundaries/failure modes.

### 3.1 Scaling Properties (S4.2, S4.3)
- **S4.2 (Timescale)**: Write a script to freeze data, perturb activations radially by +20%, and measure $u$ decay under continued training to fit the exponential $\tau_{rad} = d/(2\lambda)$.
- **S4.3 (Width Scaling)**: Run MLPs of widths $\{256, 512, 1000, 2048\}$ with the best $\lambda$ to verify the $\sqrt{d}$ target holds across scales.

### 3.2 Second Benchmarks (S5)
- **S5.1 (Rotating MNIST)**: Implement continuous rotation dataset. Run 2-layer MLP. Re-run S3 correlation.
  - *Kill Criterion*: The correlation must replicate here, otherwise A1 is not a general claim.
- **S5.2 (Split CIFAR-100)**: Implement a 4-layer ConvNet (no BatchNorm) and Split CIFAR dataset. Run BP, LN+L2, CBP, RS(best) arms.
- **S5.3 (Activation Functions)**: Run Permuted MNIST with standard ReLU vs Leaky ReLU($\alpha=0.01$) to check stability differences.

### 3.3 Limits and Failure Modes (S6)
- **S6.1 (LR Robustness)**: Sweep learning rates $\{1e-3, 1e-2, 1e-1\}$.
- **S6.2 (Optimizer Interaction)**: Run AdamW vs SGD to check if preconditioning breaks the dynamics.
- **S6.3 (Stability Tax)**: Log accuracy on all previous tasks at boundaries for BP, RS, CBP, L2-init to plot the plasticity/stability frontier.

---

## Final Review
Before presenting the final output, all comparisons will be processed to include paired test statistics, p-values, n, sign splits, and error bars on plots. All anomalous findings will be documented in `results/ANOMALIES.md` instead of being suppressed.
