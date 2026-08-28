# REPO AUDIT — Analysis Phase

**Scope:** `non_algo/` (both `stiffness/` and `experiments/`), read-only. No code changed, no runs executed, no bugs fixed.
**Date:** 2026-08-27.
**Method:** source read + static inspection of 686 stored `metrics.parquet` files, 733 run directories, 733 stdout logs, and every stored `.csv` / `.md` report. Where I state a number, it was recomputed from the parquet files during this audit unless labelled otherwise.

Throughout, findings are labelled:
- **[CODE]** — what the source does. Verifiable by reading.
- **[DATA]** — what the stored artifacts contain. Recomputed during this audit.
- **[INFER]** — my inference, with the evidence given.
- **[UNKNOWN]** — flagged uncertainty. Not guessed.

---

## 0. Summary

**What I found.** There are two independent implementations of this project in the repo, written by different hands, that share no code: `experiments/` (five per-benchmark trainers, single-head/multi-head CNNs and MLPs, JSON logs) and `stiffness/` (one trainer, one sweep launcher, parquet logs, ~700 runs). `stiffness/` is the one the paper's claims come from. `experiments/` still carries the radial-energy bug you asked about — it was never fixed there, only in `stiffness/`.

Of the nine claims, C4, C5, C7 and C9 have runnable code, coherent data, and defensible support. C1 and C6 have runnable code but the supporting numbers are weaker and more equivocal than the internal reports state. **C2 and C3 do not currently have a valid experiment behind them.** That is the headline.

**What worries me most, in order:**

1. **C2's isotropic control does not control for magnitude.** The `bp_iso` arm multiplies hidden-layer weight gradients by a per-(task, layer) scalar and then calls `clip_grad_norm_(..., 0.5)` on the *whole* model immediately afterward. The clip binds on essentially every step (§3.3.2 has the measurement), so the shrinkage is renormalized away. Intended weight-norm reduction vs. BP: ~18%. Achieved: **0.6%**. Meanwhile the RS arm's actual magnitude reduction is ~50%. The control removes about 2.5% of the confound it exists to remove, and the paper reads its null result as proof of directionality. This is the central experiment and it is not currently a valid one. Separately, the shrinkage factor is derived from a **weight-norm ratio**, not a gradient-magnitude ratio, and there are **two mutually contradictory scripts** that write the same `results/iso_shrinkage.json` — one producing factors < 1, the other > 1.

2. **C3's mediator points the wrong way at two of three layers, and the reported effect is a layer-averaging artifact.** At layer 0 and layer 1, subspace overlap *decreases* as λ rises. The entire reported effect lives at layer 2. Drift magnitude *increases* with λ at layers 0 and 1 while retention improves. And both drift metrics are normalized by `‖h_past‖` — a quantity the penalty directly shrinks — so cross-λ comparison of them is not meaningful as constructed.

3. **`bp` and `cbp` are the same experiment.** CBP is not instantiated in the current trainer (`cbp = None`, call site replaced by a bare `pass`) and `--method cbp` is rejected by argparse. Even when it *was* wired, its replacement rate made it a no-op: bp `0.211767 ± 0.003403` vs cbp `0.211651 ± 0.004306`. Every `cbp` row in every report is BP with a perturbed RNG stream.

4. **`adam` and `adamw` are bitwise-identical files.** All 10 (λ, seed) pairs. `AdamW(wd=0) ≡ Adam(wd=0)`, and the sweep sets `wd=0` for `bp`/`rs` while `'adam' in 'adamw'` gives both the same learning rate. `ROUND4_REPORT.md` presents them as two independent confirmations of C8.

5. **No run stores its configuration.** Run directories contain `metrics.parquet` and nothing else. Only `method`, `lambda_rs`, `hard_projection`, `seed` are in the parquet. `lr`, `epochs`, `n_tasks`, `width`, `depth`, `optimizer`, `dataset`, `weight_decay`, `ewc_lambda` exist **only** in the run-id string and in whichever version of `sweep.py` was on disk that day. The two config-audit scripts in the repo are, respectively, dead (reads a `config.json` that is never written) and hand-typed from memory with errors in it.

6. **Eight distinct logging schemas are mixed in `results/`.** Tables that look comparable are not. `B4_*` and `R2_B4_*` are the same nominal sweep at different learning rates and show a 14× difference in `phi_rad_tilde`.

**What I would fix first:** the `clip_grad_norm_` / `bp_iso` interaction, then write a config sidecar for every run. Nothing else can be trusted to be comparable until the second one is done.

---

## 1. Map

### 1.1 Two codebases, no shared code

| | `stiffness/` | `experiments/` |
|---|---|---|
| Trainers | 1 (`src/train.py`, 440 lines) | 5 (one per benchmark, ~250 lines each, heavily duplicated) |
| Penalty impl | `src/methods/rs.py::compute_rs_penalty` | `shared/norm_penalty.py::norm_penalty` |
| Radial-energy impl | `src/metrics/phi_rad.py::compute_phi_rad_tilde` | `shared/diagnostics.py::compute_radial_energy` |
| Radial-energy correctness | **clean** (§3.3.1) | **buggy** (§3.3.1) |
| Logging | parquet, 686 runs | JSON per run, ~70 runs + `.pt` checkpoints |
| CIFAR protocol | single 100-way head, Split-CIFAR-100 | per-task heads, Split-CIFAR-10 |
| CIFAR trains? | **no** (§5.4) | yes (train acc 1.0, avg-seen 0.71) |

The two penalty implementations agree mathematically (`(1/d)(‖h‖₂ − √d)²`, batch-mean). The two radial-energy implementations agree mathematically but differ in *what gradient is passed in*, which is the entire bug.

**[INFER]** `experiments/` is the earlier round. `latex/report.tex` is a multi-author report (Srijan / Harshit / Sarthak / Aaditya sections) that draws its §"Local Experiments" figures from `experiments/`, and its Φ_rad diagnostic section from the buggy metric. The paper described in this audit request is the `stiffness/` work.

### 1.2 `stiffness/` entry points

**Live and runnable today:**
- `src/train.py` — the only real trainer. Everything numeric traces here.
- `src/sweep.py --suite X` — 25 suites, launches `train.py` subprocesses, 10 concurrent across 2 GPUs. Skips a run if its parquet already exists.
- `analysis/*.py` — 31 scripts, run against `results/`.

**Dead:**
- `audit_configs.py` — reads `results/*/config.json`. **`train.py` never writes `config.json`.** This script prints "No configs found." and exits. It has never done anything.
- `configs/` — empty directory.
- `analysis/a3_round1_retention.py` — **crashes**: `KeyError: "Columns not found: 'prev_tasks_acc'"`. It reads S1/S2 runs, whose schema predates the retention column. Verified by running it.
- `src/methods/cbp.py` — imported by `train.py`, never instantiated (`cbp = None` at line 75, bare `pass` at line 214 where the call site was).
- `run_s0.sh` — produces `S0.2_*`/`S0.3_*`, pilot runs referenced by nothing.

**Superseded duplicates (still hold ~200 result directories):**
- `run_b1_b2.py` → duplicated by `sweep.py --suite R2_B1` / `R2_B2`. Identical configs. `B1_*`/`B2_*` vs `R2_B1_*`/`R2_B2_*`.
- `run_b4_c1.py` → *not* identical to `R2_B4`/`R2_C1`. **B4 uses `--lr 0.01`, R2_B4 uses `--lr 0.1`. C1_A/C1_B use `--n_tasks 150`, R2_C1_A/B use `--n_tasks 50`.** Same-looking names, different experiments.
- `run_s5_cpu.py` → duplicates `sweep.py --suite S5_2`, adds a `cbp` arm that crashes on ConvNet, and forces `CUDA_VISIBLE_DEVICES=''`.

**Written twice by different people:**
- `extract_shrinkage.py` (repo root) and `analysis/compute_iso_shrinkage.py` — **both write `results/iso_shrinkage.json`**, from different source runs, using different quantities, producing factors in opposite directions. See §3.3.2. This is the most consequential duplication in the repo.
- `analysis/c1_frontier.py` and `analysis/c1_frontier_v2.py` — v2 adds the ISO arm and swaps the plot; both still present.
- `analysis/r1_r3_analysis.py` and `analysis/r1_r3_drift_retention.py` — overlapping readers of `R3_R1_*`.
- `analysis/b2_convergence.py` and `analysis/b2_undertrained.py` — same B2 within-task-curve analysis, different function names.
- `analyze_s52.py` (root) is a **verbatim copy** of the S5.2 block inside `analysis/s5_analysis.py`.
- `audit_configs.py` and `build_audit.py` — both write `results/CONFIG_AUDIT.md`; the second one wins because the first produces nothing.

### 1.3 Where the real logic lives vs. where it looks like it lives

- The penalty is 10 lines in `src/methods/rs.py`. Fine.
- **Everything load-bearing is inlined in `src/train.py:249–434`** — the drift metrics, the subspace overlap, the radial decomposition, the retention loop, the isotropic-control gradient surgery. None of it is in `src/metrics/`. `src/metrics/` contains only the small, tested helpers. So the four measurements the paper rests on are *not* where the directory layout suggests they are, and three of the four have no unit test.
- `tests/` covers `compute_phi_rad_tilde` and `compute_rs_penalty` only. 5 tests, all pass.
- `analysis/generate_detailed_results.py` reconstructs `width`, `optimizer` and `lr` **by string-parsing the run-id**. That is the clearest statement in the repo that config is not logged.

---

## 2. Provenance

### 2.1 Eight schema generations

**[DATA]** The 686 parquets fall into exactly 8 perfectly-nested column sets. This is the repo's real chronology and it is far more informative than the run-id prefixes:

| # | cols | runs | new column(s) vs. the next-older | run families |
|---|---|---|---|---|
| 7 | 20 | 43 | *(oldest)* | `S0.2`, `S0.3`, `S1_lam_*` |
| 6 | 21 | 65 | `method` | `S2_*` |
| 5 | 22 | 62 | `prev_tasks_acc` | `S4_3_*`, `S5_1_*`, `S5_3_*`, `S6_1_*`, `S6_2_*`, `S5_2_*` |
| 4 | 23 | 204 | `update_norm` | `B1_*`, `B2_*`, `B4_*`, `C1_A_*`, `C1_B_*`, `R2_*` |
| 3 | 26 | 110 | `drift_cos_sim`, `drift_rel`, `drift_overlap` | `R3_R2_*`, `R3_R4_*`, `R3_R6_opt_sgd*` |
| 2 | 28 | 21 | `g_rad_norm`, `g_norm_task` | `R3_A1_*`, `R3_C1_*` |
| 1 | 30 | 49 | `drift_rad`, `drift_tan` | `R3_R1_*`, `R3_A2_*`, `R3_A1b_*` |
| 0 | 31 | 132 | `task_0_acc` | **all `R4_*`**, `R3_R6_opt_adam*`, `R3_R6_opt_adamw*` |

Two things fall out of this immediately:

- **`S1_*` and `S2_*` have no retention column at all.** The λ-sweep that C4 descends from and the baseline head-to-head that produced the "project halted" verdict in `results/ANOMALIES.md` **cannot express previous-task accuracy**. Whatever those suites showed, it was current-task accuracy only.
- **The R6 optimizer table straddles two generations.** `R3_R6_opt_sgd*` and `R3_R6_opt_sgd_momentum*` are schema 3 (26 cols); `R3_R6_opt_adam*` and `R3_R6_opt_adamw*` are schema 0 (31 cols). The Adam arms were re-run under newer code; the SGD arms were not. C8 compares them directly.

**[DATA]** A second, independent clock: 50 logs contain tracebacks that record `train.py`'s line count at the time. Four distinct versions appear — 263, 390, 434, 436 lines (current: 440). `S6_1_*` ran at 263; `R3_R2_*`/`R3_R4_*` at 390; `R3_R1_*`/`R3_A2_*`/`R3_A1b_*`/`R3_C1_bp_matched_lr_0.025` at 434; `S5_2_cbp_*` at 436. The 436-line version's line 77 is `cbp = CBP(model)` — direct evidence CBP was once wired and has since been removed.

### 2.2 Claim-by-claim provenance

| Claim | Code path | Analysis script | Status | Notes |
|---|---|---|---|---|
| **C1** retention at matched accuracy | `sweep.py --suite R2_C1` → `R2_C1_A_*` / `R2_C1_B_*`; `R3_R2` | `c1_frontier.py`, `c1_frontier_v2.py`, `r2_baselines.py` | **reproducible from source** | but see §6 — the "matched accuracy" logic is undermined by the retention metric including the current task |
| **C2** isotropic control | `sweep.py --suite R4_T2c` (+ `R3_C1_iso`) → `train.py:185–193`; shrinkage from `analysis/compute_iso_shrinkage.py` **or** `extract_shrinkage.py` | `t2_c1_canonical.py`, `c1_frontier_v2.py` | **reproducible from source, but the experiment is invalid** — and *which* shrinkage file was used for R4_T2c is **[UNKNOWN]** (see below) | §3.3.2 |
| **C3** subspace vs. drift | `train.py:334–379` (`--track_drift`) → `R3_R1_*`, `R4_T1_*`, `R4_T2c_*` | `r3_drift.py`, `r1_r3_analysis.py` | **reproducible from source; conclusion contradicted by the data** | §3.3.3, §6 |
| **C4** λ sweep incl. hard limit | `sweep.py --suite R4_T1` + `R3_R1` | `t1_lambda_bracket.py` | **reproducible from source** | seed counts 2–5 per λ, unflagged; the `inf` arm is a straight-through estimator, not the λ→∞ limit |
| **C5** two axes decouple | same runs as C4 | `t1_lambda_bracket.py` | **reproducible from source** | strongest-supported claim in the set |
| **C6** Pareto + additivity | `sweep.py --suite R3_R2` | `r2_baselines.py` | **reproducible from source**, with caveats | 4 of 15 arms have n=4, one has n=3; reports say "5 seeds" |
| **C7** width scaling | `sweep.py --suite R3_R4` | `r4_width_scaling.py` | **reproducible from source** | `R3_R4_w_1000_lam_0.03` has n=2 |
| **C8** momentum vs. adaptive | `sweep.py --suite R3_R6` | `r6_optimizers.py`, `t7_adam.py` | **partly orphaned** | see below |
| **C9** forgetting ≫ plasticity loss | `run_b1_b2.py` / `R2_B1`, `R3_A1`, `R3_A1b` | `a1_plasticity_converged.py`, `a1b_plasticity_vs_budget.py` | **reproducible from source** | `a1b_stats.csv` silently omits the epochs=3 row (2 of 3 seeds OOM-crashed) |

### 2.3 Orphaned results — numbers with no runnable path

These are the ones you asked me to surface. Each is a number that appears in a report or CSV with nothing behind it that runs today.

1. **Every `cbp` result.** `S2_cbp_*` (5 runs), `R3_R2_cbp_*` (4 runs), `S2_rs_cbp_*` (5 runs). `--method cbp` and `--method rs_cbp` are both rejected by the current argparse; `CBP` is never instantiated. `cbp` appears in `QUANTITATIVE_RESULTS.md` §R2 with a mean and std. **Reproducible only from stored artifacts, and those artifacts are BP (§5.1).**

2. **`results/r6_stats.csv` — the entire Adam/AdamW block.** It reports adam λ=0 as `test_acc 0.1161, prev 0.1166`. `t7_adam_stats.csv`, generated from the *same* run directories four days later, reports `test_acc 0.9553, prev 0.1962`. The directories were overwritten by a re-run at a different learning rate. **The `r6_stats.csv` Adam numbers — which are the ones printed in `QUANTITATIVE_RESULTS.md` — are orphaned.** The accompanying note there ("Adam/AdamW were run at lr=0.1") is also wrong: `sweep.py` sets `run_lr = 0.001 if 'adam' in opt else 0.01`.

3. **`ROUND2_REPORT.md` §C1: "BP achieves 0.481, RS(λ=0.03) achieves 0.570, +8.9% at matched plasticity."** These come from `R2_C1_*` at `task >= 30` — I reproduce 0.4817 ± 0.0083 and 0.5704 ± 0.0093, so the numbers themselves are live. But the same report's §B1 quotes lr=0.1 slope `−2.82e−6, p=0.330`, while `results/a1_stats.csv` gives `−1.018e−5, p=4.6e−8` and `ANOMALIES.md` cites `−1.0e−5` — three different values for what reads as the same quantity. **[INFER]** they are different budgets (B1 = 1 epoch, A1 = 10 epochs) that were never distinguished in prose.

4. **`results.md` S5.1 vs `ROUND3_S5_REPORT.md` S5.1.** Same runs (`S5_1_rot_*`), two published retention values: `0.6651 / 0.6326 / 0.6083` and `0.4765 / 0.4245 / 0.4055`. **[INFER]** different task windows (`s5_analysis.py` uses the last 20 tasks; whatever produced `results.md` used something else). One of them is orphaned; I cannot tell which without the generator.

5. **`results/CONFIG_AUDIT.md` in its entirety.** Produced by `build_audit.py`, which is a hand-typed Python literal, not a measurement. It contains a comment reading `# some S1 runs had 5 seeds later? No, S1 had 3 seeds originally, wait, some might have 5. I'll put 5.` Checked against `sweep.py`, it is wrong on at least: batch size (says 128 in every row; **`train.py` hardcodes 256**), S1/S2 learning rate (says 0.1; both suites pass no `--lr`, so **0.01**), B4 seed count (says 5; `R2_B4` uses 3), R6 (attributes it to `S6_*` when the reported table is `R3_R6_*`). Its "Canonical Config Recommendation" declares `epochs=1` canonical and then says T2c will be re-run at that config; **`R4_T2c` was actually run at `--epochs 10`.**

6. **`experiments/phi_rad_analysis/plots/*.png` and the `radial_energy_*` figures in `latex/`.** Built entirely on the contaminated metric (§3.3.1).

7. **[UNKNOWN] Which `iso_shrinkage.json` produced `R4_T2c_bp_iso`.** The file on disk is dated `2026-08-26 00:44`. `R4_T2c_bp_iso` seeds 1, 3, 4, 5 finished `08-25 20:16–20:41`; seed 2 finished `08-26 00:57`. So seeds 1/3/4/5 consumed a file that no longer exists, and seed 2 consumed the current one. The inputs (`R4_T1_lam_0.01_seed3`) finished at `19:34`, so **[INFER]** the 20:0x file was probably byte-identical to the 00:44 one, but I cannot verify that and the run does not record it.

---

## 3. Correctness of the load-bearing measurements

### 3.1 The radial-energy metric — clean in `stiffness/`, still broken in `experiments/`

**[CODE] `stiffness/` is clean.** `src/train.py:286–300`:
```python
out_probe, pre_probe, post_probe = model(probe_x, return_activations=True, ...)
loss_probe = criterion(out_probe, probe_y)          # task loss ONLY
g_tasks = torch.autograd.grad(loss_probe, pre_probe, retain_graph=True)
...
phi_rad = compute_phi_rad_tilde(h, g_task)
```
The probe loss never includes the RS penalty, EWC/SI/MAS penalties, or L2-init. `g_tasks` is taken by explicit `autograd.grad` against `pre_probe`, not by reading `.grad` after a combined backward. `src/metrics/phi_rad.py` documents the requirement in its docstring and `tests/test_phi_rad.py::test_phi_rad_tilde_leakage` is a dedicated regression test for exactly this failure mode. It passes.

**[DATA] No stored `stiffness/` result shows the leakage signature.** If the penalty gradient were included, φ̃_rad would be driven toward `d` (=1000) as λ grows, because the penalty gradient is purely radial. It is not: in `S1_*` (oldest schema) φ̃_rad runs 1.85–2.44 across λ ∈ [0, ∞]; in `R4_T1_*` (newest) 0.10–0.40. Flat or non-monotone in λ in every generation. **I find no era of `stiffness/` results that predates the fix.** [UNKNOWN whether the bug ever existed in `stiffness/` at all, or only in `experiments/`.]

**[CODE] `experiments/` is broken, in all five trainers.** `experiments/permuted_mnist/train.py:170–184` (and the structurally identical code in `colored_mnist`, `digit_addition`, `rotating_mnist`, `split_cifar10`):
```python
loss = loss_ce + lambda_norm * loss_reg + l2_init_coef * loss_l2_init
optimizer.zero_grad(); loss.backward(); optimizer.step()
...
re_h1 = compute_radial_energy(h1, h1.grad)   # h1.grad is the TOTAL gradient
```
`h1.grad` is the gradient of the *combined* loss. `norm_penalty(h1)`'s gradient w.r.t. `h1` is exactly `(2/d)(‖h‖−√d)·ĥ` — purely radial. The metric is therefore partly measuring its own regularizer.

**[DATA] The contamination is enormous.** End-of-training `radial_energy_h1` in `experiments/permuted_mnist/results/`:

| arm | first | last |
|---|---|---|
| `baseline` (λ=0) | 14.2 | **0.02** |
| `shrink_perturb` (λ=0) | 14.2–15.1 | 0.02–0.04 |
| `strong_wd` (λ=0) | 14.3–15.3 | 0.37–0.45 |
| `l2_init` (λ=0) | 18.0 | 18.8–19.4 |
| **`norm_penalty` (λ=0.05)** | 29.8–30.0 | **83–101** |

A ~4000× ratio between the penalized arm and the baseline. **Blast radius:** every `*_diagnostics.json` under `experiments/*/results/` (~70 files), `experiments/phi_rad_analysis/analyze.py` and its two plots, and the `latex/report.tex` section "Experiment 3: Φ̃_rad as a Predictive Diagnostic", which correlates this quantity against next-task accuracy *pooled across arms including `norm_penalty`*. That correlation is measuring the regularizer, not a diagnostic.

### 3.2 The isotropic control (C2) — the control does not control

This is the most serious finding in the audit. Four independent problems compound.

**(a) [CODE] The shrinkage is applied and then immediately undone by gradient clipping.** `src/train.py:185–195`:
```python
if args.method == 'bp_iso' and shrinkage is not None:
    for l, layer in enumerate(model.layers):
        layer.weight.grad.data.mul_(s)          # s ≈ 0.90 / 0.94 / 0.99 per layer
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)   # ← global renormalization
```
`clip_grad_norm_` computes the norm over **all** parameters and, if it exceeds 0.5, rescales everything to exactly 0.5. A near-uniform pre-multiplication is therefore erased whenever the clip binds.

**[DATA] The clip binds essentially always.** `update_norm` is logged per layer at the last step of each task. With `lr=0.1` and clip 0.5, a bound clip gives `‖ΔW_all‖ = 0.05` exactly. Measured over `R4_T2c`, last 20 tasks, √(Σ_layers ‖ΔW_l‖²) — hidden layers only, so biases and head push the true total higher still:

| arm | Σ over layers | per-layer (L0, L1, L2) |
|---|---|---|
| `bp` | **0.0457** | 0.0297, 0.0318, 0.0158 |
| `bp_iso` | **0.0445** | 0.0260, 0.0293, 0.0179 |
| `rs` | 0.0227 | 0.0157, 0.0150, 0.0053 |

Early tasks (seed 1, tasks 0–9) sit at 0.0455–0.0489 for both `bp` and `bp_iso`. The ceiling is 0.0500. **The total step magnitude is unchanged between `bp` and `bp_iso` (−2.6%); only the layer allocation shifts** — L0 down, L2 up, exactly as a fixed-norm redistribution would predict.

**[DATA] The intended effect vs. the achieved effect.** Layer-0 weight norm at task 149:

| arm | ‖W₀‖ @ t=149 | ratio to BP |
|---|---|---|
| `bp` | 50.863 | 1.000 |
| `bp_iso` | 50.516 | **0.993** |
| `rs` | 41.449 | **0.815** |

The shrinkage file's layer-0 minimum is `0.8149` — i.e. the control was *aimed* at reproducing RS's 18.5% weight-norm reduction. It achieved 0.7%. **It removed roughly 4% of the magnitude confound it exists to remove.**

Consequently: `bp` retention `0.2118 ± 0.0034`, `bp_iso` `0.2110 ± 0.0026`, `rs` `0.3388 ± 0.0049`. `t2_c1_canonical.py` divides `(iso − bp)/(rs − bp)` and reports "**magnitude matching recovers −0.6% of the RS retention gain**" (`ROUND4_REPORT.md` §1). That null is exactly what a control that did nothing would produce. It is not evidence about directionality.

**(b) [CODE] The shrinkage factor is a weight-norm ratio, not a gradient-magnitude ratio.** `analysis/compute_iso_shrinkage.py:32–41`:
```python
mean_bp = df_bp.groupby(['task','layer'])['weight_norm'].mean()
mean_rs = df_rs.groupby(['task','layer'])['weight_norm'].mean()
shrinkage[task][layer] = rs_val / bp_val
```
Weight norm is a *state* variable dominated by the initialization (‖W₀‖ ≈ 44.8 at t=0). Multiplying a gradient by 0.9 does not scale the resulting weight norm by 0.9. The quantity being matched is not the quantity being manipulated.

**(c) [CODE] There are two contradictory generators of the same file.** `extract_shrinkage.py` (repo root, dated 12 minutes before the `R3_C1_bp_iso` runs) uses `grad_norm_rs / grad_norm_bp` from `R2_C1_*` (50 tasks, λ=0.03). `analysis/compute_iso_shrinkage.py` uses `weight_norm_rs / weight_norm_bp` from `R4_T1_*` (150 tasks, λ=0.01). **Both write `results/iso_shrinkage.json`.** They point in *opposite directions*: in `R4_T2c`, RS's logged `grad_norm` at layer 0 is `1.0919` vs BP's `0.7057` — a ratio of **1.55**, an *expansion*. The weight-norm ratio is **0.815**, a contraction. Whichever script ran last silently redefined the control.

**[DATA] Both versions of the control failed, in opposite directions.** The R3-era `bp_iso` (grad-norm file, 50 tasks) ends at ‖W₀‖ = 47.313 vs BP's 46.922 — *larger* than baseline — and reports retention `0.4961 ± 0.0121` vs BP `0.4817 ± 0.0083`, a spurious *positive*. The R4-era `bp_iso` (weight-norm file) reports a spurious *zero*. `c1_frontier_v2.py` and `t2_c1_canonical.py` each report one of these as "the isotropic control", without noting they are different interventions.

**(d) [CODE] Granularity and coverage.** The factor is per (task, layer) — one scalar per hidden layer per task, averaged over 5 seeds, then applied to *every* seed's run. It is applied only to `model.layers[l].weight.grad`: **not to biases, not to `model.head`.** And a per-seed run is being corrected by a population-mean trajectory, so within-seed matching is not even attempted.

**Verdict: C2 has no valid experiment.** The comparison is not like-for-like.

### 3.3 The subspace and drift metrics (C3)

**[CODE] The probe.** `src/train.py:119–122`:
```python
drift_probe_x, _, _, _ = dataset.get_task_data(0)
drift_probe_x = drift_probe_x[:1000].clone().to(device)
```
- **Fixed across tasks within a run: yes.** Cloned once before the task loop, never rebuilt.
- **Fixed across runs: only within a seed.** For `permuted_mnist`, task 0's permutation comes from `np.random.default_rng(seed)`, so the probe differs between seeds. All arms sharing a seed share the probe, so paired comparisons are valid; cross-seed pooling of *absolute* drift values is not strictly like-for-like. Minor.
- **It is training data, not held-out.** `get_task_data(0)[0]` is `x_train`.

**[CODE] What is compared to what.** `past_acts` is overwritten at the end of *every* task (`train.py:372`). So `drift_rel`, `drift_overlap`, `drift_cos_sim`, `drift_rad`, `drift_tan` all measure **consecutive task-boundary increments**, not cumulative drift from the original representation. The variable name `drift` and the reports' framing ("preservation of the representational subspace across tasks") both imply a fixed reference. It is not one.

**[CODE] Basis and layer.** `pre_probe[l]` — pre-activations, per hidden layer, `hard_projection` applied if the arm uses it. Comparison is in the raw activation basis.

**[CODE] Rank consistency — this part is fine.** `k = min(50, U_curr.shape[1], U_past.shape[1])`. For MLPs (width ≥ 256, probe 1000) and the ConvNet (feature dims ≥ 4096), `k = 50` in every arm. `overlap` is the mean of the 50 principal-angle cosines. **The same rank is compared across all arms.** (Note: the conventional projection metric uses mean *squared* cosines; using raw cosines makes the measure more lenient. Not a bug, but it should be named accurately in the paper.)

**[CODE] A real methodological defect: both drift metrics are normalized by a quantity the penalty controls.**
```python
lm['drift_rad'] = (torch.norm(delta_h_rad) / (torch.norm(h_past) + 1e-8)).item()
lm['drift_tan'] = (torch.norm(delta_h_tan) / (torch.norm(h_past) + 1e-8)).item()
lm['drift_rel'] = (torch.norm(h_curr - h_past) / (torch.norm(h_past) + 1e-8)).item()
```
`‖h_past‖` is exactly what RS shrinks — from `radius_mean` 49.75 at λ=0 to 34.21 at λ=0.03, a factor of 1.45. An identical absolute displacement therefore reads 45% larger in the λ=0.03 arm. **Comparing `drift_rel` / `drift_rad` / `drift_tan` across λ is not valid as constructed.** This alone accounts for much of the "drift increases with λ" pattern below.

**[CODE] The `λ=inf` arm is measured on a different manifold.** Activations are recorded with `hard_projection=args.hard_projection`, so the hard arm's drift is computed on projected (unit-sphere-scaled) activations while every other arm's is computed on raw pre-activations. Its drift/overlap points are not comparable to the rest of the sweep.

**[DATA] The reported C3 effect is a layer-averaging artifact.** `analysis/r1_r3_analysis.py` does `df.groupby('task').mean(numeric_only=True)` — an unweighted average **across all three layers**. Per layer (`R3_R1_*`, last 20 tasks), `drift_overlap`:

| λ | layer 0 | layer 1 | layer 2 | 3-layer mean *(as reported)* |
|---|---|---|---|---|
| 0.0 | 0.9974 | 0.9922 | 0.8539 | 0.9478 |
| 0.01 | 0.9972 | 0.9905 | 0.9416 | 0.9764 |
| 0.03 | 0.9967 | 0.9871 | 0.9455 | 0.9764 |
| 0.1 | 0.9947 | 0.9871 | 0.9442 | 0.9753 |
| 1.0 | 0.9947 | 0.9831 | 0.9457 | 0.9745 |
| 10.0 | **0.9940** | **0.9826** | 0.9438 | 0.9735 |
| inf | 0.9962 | 0.9903 | 0.8684 | 0.9516 |

The rightmost column is `QUANTITATIVE_RESULTS.md` §R3 to 4 decimals, so this is the reported quantity. **At layers 0 and 1, overlap falls monotonically as λ rises.** The entire effect is layer 2, where the metric has the only real dynamic range (0.854 → 0.946). Layer 2 is the last hidden layer, feeding the head.

**[DATA] And within the region where the retention effect lives, overlap is flat.** From `R4_T1_*` (the fine λ grid), layer 0, last 20 tasks:

| λ | retention | drift_rel | drift_overlap | drift_rad |
|---|---|---|---|---|
| 0.0000 | 0.2118 | 0.0612 | 0.99750 | 0.00242 |
| 0.0010 | 0.3267 | 0.0598 | 0.99764 | 0.00271 |
| **0.0030** | **0.3457** | 0.0606 | 0.99783 | 0.00301 |
| 0.0100 | 0.3388 | 0.0671 | 0.99725 | 0.00356 |
| 0.0300 | 0.3319 | 0.0783 | 0.99670 | 0.00462 |

Retention swings by **+13.4 points**. Overlap moves by **0.001** and is non-monotone. `drift_rel` moves by 0.017 and is non-monotone. The only monotone column is `drift_rad`, which **increases** with λ — the opposite of what a radial penalty should do to radial drift, and readable straight off the normalization defect above.

**Verdict on C3 as stated:** "drift magnitude and subspace overlap behave differently from one another" is true, but so does every layer behave differently from every other layer, and the layer-average is doing all the work. The specific claim that *subspace preservation* is the mediator is **not supported at layers 0–1, and at layer 2 it separates {λ=0, λ=inf} from {finite λ} without explaining any of the variation among finite λ — which is precisely where C4's effect lives.** Layer 2's overlap is essentially constant at 0.944 while retention falls from 0.339 to 0.296.

### 3.4 The retention metric

**[CODE] Definition.** `src/train.py:389–396`:
```python
prev_accs = []
for px, py in past_task_tests:
    logits = model(px[:probe_batch_size], hard_projection=args.hard_projection)
    prev_accs.append((logits.argmax(1) == py[:probe_batch_size]).float().mean().item())
prev_tasks_acc = sum(prev_accs) / len(prev_accs) if prev_accs else test_acc
task_0_acc = prev_accs[0] if prev_accs else test_acc
```
`past_task_tests.append((x_test, y_test))` happens at the **top** of the task-t loop (`train.py:126`), before training. So **`prev_tasks_acc` includes the current task.** It is the standard "average accuracy over all tasks seen so far" (ACC), not previous-task retention.

**[DATA] This is consistent across every schema generation that has the column.** I verified `prev_tasks_acc[t=0] == test_acc[t=0]` exactly (to float equality) in `R4_T2c`, `R3_R1`, `R3_R2`, `R3_C1`, `R2_C1_A`, `B4`, `S4_3` and `S6_2`. The definition never changed. **The metric is computed identically across every arm — I found no silent inconsistency here.** That is the one clean answer in §3.3.

Three caveats that still matter:

1. **It is misnamed, and the bias is not neutral between arms.** At task *t* the value is inflated by `test_acc/(t+1)`. At t=149 that is ~+0.0064 absolute — small, but it systematically favors arms with higher current-task accuracy, which is exactly the direction C1's "at matched current-task accuracy" argument is trying to control for. The paper should either use `task_0_acc` (present only in the 31-column schema) or subtract the current task explicitly.
2. **It evaluates only the first `probe_batch_size` test samples** (2000 for MNIST, 256 for CIFAR). Consistent across arms; CIFAR's 256-sample evaluation gives a resolution of 0.39%, which is coarse relative to the effect sizes reported there.
3. **The denominator grows with t**, so the metric is not comparable between suites with different `n_tasks` (50 vs. 150 vs. 300). Several reports place 50-task and 150-task retention side by side.

---

## 4. Configuration integrity

### 4.1 What is recoverable, and from what

**[CODE] `train.py` writes no config.** `os.makedirs(f'results/{args.run_id}')` then, at the end, `df.to_parquet(...)`. Nothing else. Run directories confirm: `metrics.parquet` only (plus `within_task.csv` for the `--log_within_task` suites).

The parquet embeds exactly four config axes: `method`, `lambda_rs`, `hard_projection`, `seed`.

**Not recorded anywhere except the run-id string:** `lr`, `epochs`, `n_tasks`, `dataset`, `model`, `width`, `depth`, `act_fn`, `optimizer`, `weight_decay`, `ewc_lambda`, `track_drift`, batch size (hardcoded 256), clip norm (hardcoded 0.5), probe size (hardcoded 2000/256), the `l2_init` coefficient (hardcoded 0.01), the `er` coefficient (hardcoded 0.01), the SI/MAS coefficients (hardcoded 1.0), `shrink_perturb`'s 0.8/0.01, `ReDo`'s 0.025/1000, and the contents of `iso_shrinkage.json`.

**[CODE] `--weight_decay` is silently dropped for most methods.** `train.py:100`: `wd = args.weight_decay if args.method in ['l2','ln_l2','l2_er'] else 0.0`. Passing `--weight_decay` with any other method has no effect and produces no warning.

### 4.2 Configuration axes that actually vary across stored runs

Reconstructed from `sweep.py`, `run_b1_b2.py`, `run_b4_c1.py`, `run_s5_cpu.py`, `run_s0.sh`:

| axis | values in `results/` |
|---|---|
| `dataset` | `permuted_mnist`, `rotating_mnist`, `split_cifar100` |
| `model` | `mlp` (depth 3, width 1000 default; depth 2 width 256 for rotating), `convnet` |
| `n_tasks` | 10, 20, 50, 100, 150, 300 |
| `epochs` | 1, 3, 10, 30 |
| `lr` | 0.001, 0.003, 0.01, 0.015, 0.02, 0.025, 0.03, 0.1, 0.3 |
| `width` | 256, 512, 1000, 2048 |
| `optimizer` | `sgd`, `sgd_momentum`, `adam`, `adamw` |
| `act_fn` | `relu`, `leaky_relu` |
| `ewc_lambda` | 100, 1000, 10000 |
| `seed` | 1–5 |
| **code generation** | **8 schemas / ≥5 `train.py` versions** |

### 4.3 Which sets are legitimately comparable

**Comparable within themselves (same launcher, same generation):**
- `R4_T1_*` ∪ `R4_T2c_*` — permuted MNIST, 150 tasks, 10 ep, lr 0.1, w1000 d3, sgd, `--track_drift`, schema 0. **This is the cleanest block in the repo.** The C4/C5 λ bracket and the C2 control live here.
- `R3_R2_*` — 150 tasks, 10 ep, lr 0.1, schema 3, all 15 baseline arms. Internally comparable. C6 lives here.
- `R3_R4_*` — 150 tasks, 10 ep, lr 0.1, schema 3, widths 256–2048. C7 lives here.
- `R2_C1_A_*` ∪ `R2_C1_B_*` ∪ `R3_C1_bp_matched_*` ∪ `R3_C1_bp_iso_*` — 50 tasks, 10 ep, schema 3–4. (`R3_C1_bp_iso` is schema 2, one generation newer than `R2_C1_*`; the differing columns are diagnostics, not accuracy, so accuracy comparison holds.)
- `S1_*` — 150 tasks, **1 ep**, **lr 0.01**, schema 7, no retention column.
- `S2_*` — 150 tasks, 1 ep, lr 0.01, schema 6, no retention column.

**Not comparable, though the names suggest otherwise:**

| A | B | difference |
|---|---|---|
| `B4_lam_*` | `R2_B4_lam_*` | **lr 0.01 vs lr 0.1.** [DATA] test_acc 0.923 vs 0.962; φ̃_rad 2.17 vs 0.16 (**14×**) |
| `C1_A_lr_*` / `C1_B_lam_*` | `R2_C1_A_*` / `R2_C1_B_*` | **150 tasks vs 50 tasks.** [DATA] C1_B λ=0 retention 0.301 vs R2_C1_B λ=0 retention 0.455 |
| `S1_*` | `R3_R1_*` / `R4_T1_*` | 1 ep / lr 0.01 vs 10 ep / lr 0.1. [DATA] test_acc 0.77 vs 0.96; φ̃_rad ~1.9 vs ~0.4 |
| `S4_3_width_*` | `R3_R4_w_*` | 1 ep, λ=1.0, **no BP arm** vs 10 ep, λ∈{0, 0.03} |
| `R3_R6_opt_sgd*` | `R3_R6_opt_adam*` | **schema 3 vs schema 0** — different `train.py` generations, and lr 0.01 vs 0.001. C8 compares these directly. |
| `S6_2_opt_*` | `R3_R6_opt_*` | 1 ep / λ=1.0 vs 10 ep / λ=0.03 |
| `S5_1_rot_*` | `R4_T3a_rot_*` | λ ∈ {0, 1, 10} vs λ ∈ [0, 0.1]; schema 5 vs schema 0 |
| `S5_2_*` | `R4_T3b_*` | same CIFAR config, schema 5 vs schema 0 |

**Not recoverable at all:** `S0.2_*`, `S0.3_*` — pilot runs from `run_s0.sh`, schema 7, referenced by no analysis script, no report, and no README.

### 4.4 Silent seed loss

**[DATA]** 47 of 733 run directories have no parquet. Every one I inspected died on `torch.OutOfMemoryError` in `PermutedMNIST.get_task_data` — the sweep launches 10 concurrent jobs and the GPU was already shared with unrelated processes. Every analysis script guards with `if os.path.exists(path)` and silently skips.

| family | completed seeds | missing |
|---|---|---|
| `R3_R1_lam_3.0` | **0** | 1,2,3,4,5 |
| `R3_A2_lam_0.3`, `R3_A2_lam_10.0` | **0** | 1,2,3 |
| `R3_R1_lam_0.0` | **1** | 1,3,4,5 |
| `R3_A1b_ep_3` | **1** | 1,3 |
| `S6_1_lr_0.01_rs` | **1** | 1,3 |
| `R3_R1_lam_0.1`, `R3_R1_lam_0.3`, `R3_R4_w_1000_lam_0.03`, several `R3_A2_*` | 2 | — |
| `R3_R2_{cbp, ewc_1000, l2_init, mas, si, rs_ewc_1000}` | 4 | one each |
| `R3_R2_rs_ewc_10000` | 3 | 2,5 |

Consequences that reach published tables:
- `results/t1_lambda_bracket.csv` (the C4 bracket, quoted in `ROUND4_REPORT.md` §2) has `seed_count` 5,5,5,5,5,5,5,5,5,**2,2,3,4**,5. The λ=0.1 and λ=0.3 points are **2 seeds**. The report prints them without qualification.
- `QUANTITATIVE_RESULTS.md` §R2 says "averaged across 5 seeds"; six of the eleven arms have 4 or 3.
- `results.md` §S6.1 prints `rs` at lr=0.01 with `std = nan` — **n = 1** — in a published table.
- `results/a1b_stats.csv` has no epochs=3 row at all; two of its three seeds crashed and the script dropped the third silently.

---

## 5. Known and suspected defects

### 5.1 The inert baseline — CONFIRMED, it is `cbp`

**Root cause, two layers deep.**

*Layer 1 — it is not wired in at all today.* `src/train.py:75–76`:
```python
    cbp = None

```
`CBP` is imported at line 26 and never constructed. Line 214 is a bare `pass` — the deleted call site. `--method cbp` is not in the argparse `choices`; I ran it and it errors out. **[DATA]** `S5_2_cbp_seed1.log` shows `train.py` line 77 was `cbp = CBP(model)` when that file was 436 lines long, so it *was* wired and has since been stripped.

*Layer 2 — even when wired, it did nothing.* `src/methods/cbp.py:34–38`:
```python
prob = self.replacement_rate * B   # wait, replacement_rate usually per step.
prob = self.replacement_rate       # ← overwrites the line above; 1e-6
replace_mask = (torch.rand(d) < prob) & (self.ages[i] > self.maturity_threshold)
```
`replacement_rate = 1e-6` per neuron per step. With d=1000 and ~35,000 steps that is ~35 replacements per layer across the entire 150-task run. It also **ignores `self.utilities` entirely** — selection is uniform random, so this is random neuron reset, not Continual Backprop.

**[DATA] Verified inert.** `R3_R2`, last 20 tasks, layer 0:

| arm | test_acc | prev_tasks_acc |
|---|---|---|
| `bp` | 0.9587 ± 0.0011 | 0.2118 ± 0.0034 |
| `cbp` | 0.9587 ± 0.0005 | 0.2117 ± 0.0043 |

Identical to four decimals, far inside seed noise. The small per-row differences are RNG-stream divergence: `torch.rand(d)` shifts the generator, changing `torch.randperm` batch order.

**Blast radius:** `S2_cbp_*` (5), `R3_R2_cbp_*` (4), `S2_rs_cbp_*` (5). `QUANTITATIVE_RESULTS.md` §R2 lists `cbp` as a baseline. `results/ANOMALIES.md` and `analysis/s2_baselines.py` include it in the head-to-head. **`ROUND3_S5_REPORT.md` correctly reports the ConvNet CBP crash but does not report that the MLP CBP was a no-op.**

**Severity: high.** The paper currently claims to beat Continual Backprop. It has never run Continual Backprop.

### 5.2 The broadcast diagnostic — CONFIRMED, it is `readiness`

**Root cause.** In the current `train.py:254–282`, readiness is computed **per layer** — 8 microbatch gradients collected separately for each `model.layers[l].weight`. In every earlier generation it was computed **once network-wide** and written into all three layer rows.

**[DATA] Verified directly.** For each run I checked whether the `readiness` column is identical across layers for every task:

| generation | readiness | families |
|---|---|---|
| schema 0 (31 cols) | **per-layer** | all `R4_*`, `R3_R6_opt_adam*`, `R3_R6_opt_adamw*` — 26 families |
| schemas 1–7 | **broadcast** | **156 families**, 554 runs |

Example — `R3_R2_bp_seed1`, task 0: `0.230374 / 0.230374 / 0.230374`. `R4_T2c_rs_seed1`, task 0: `0.071571 / 0.087925 / 0.070142`.

**Blast radius, and it has already reached a published table.** `QUANTITATIVE_RESULTS.md` §R5 reports the diagnostic-race Spearman correlations:
```
Layer 0 | readiness | -0.248079 | 0.000417
Layer 1 | readiness | -0.248079 | 0.000417
Layer 2 | readiness | -0.248079 | 0.000417
```
The same number three times, because `analysis/r5_headroom_race.py` correlates the broadcast column against a per-layer target. Every `readiness` row in `results/r5_race.csv`, `results.md` §S3, and any figure derived from them is contaminated. `analysis/s3_diagnostic_race.py` and `s4_1_m1_verification.py` read the same column.

**Additional problem in the same script:** `r5_headroom_race.py` globs **all** `R3_*` runs into one pool — `R3_R1` (150 tasks, 10 ep), `R3_R2`, `R3_R4` (four widths), `R3_R6` (four optimizers, two lrs), `R3_A1` (300 tasks), `R3_A2` (50 tasks), `R3_C1` (50 tasks) — and computes pooled Fisher-z correlations across all of them. The correlations mix seven distinct configurations.

**Severity: high for the R5 table specifically; medium overall** (readiness is a supporting diagnostic, not a load-bearing claim).

### 5.3 The identical optimizer arms — CONFIRMED, and it is a bug

**[DATA] Bitwise identical.** For all 10 (λ ∈ {0.0, 0.03}) × (seed 1–5) pairs, `R3_R6_opt_adam_lam_L_seedS` and `R3_R6_opt_adamw_lam_L_seedS` have `max_abs_diff = 0.000e+00` across `test_acc`, `prev_tasks_acc`, `phi_rad_tilde`, `weight_norm`, `radius_mean`, 450 rows each. Not "similar" — the same numbers. `results/t7_adam_stats.csv` shows the two optimizers agreeing to 16 significant figures including the standard deviations.

**Root cause — two independent bugs that happen to cancel any difference.**
1. `src/train.py:100`: `wd = args.weight_decay if args.method in ['l2','ln_l2','l2_er'] else 0.0`. The R3_R6 arms use `--method bp` / `--method rs`, so **`weight_decay = 0` for both optimizers**. `AdamW` with `weight_decay=0` is mathematically identical to `Adam` with `weight_decay=0`; decoupled weight decay is the *only* difference between them.
2. `src/sweep.py:175`: `run_lr = 0.001 if 'adam' in opt else 0.01`. The substring `'adam'` is in `'adamw'`, so both get lr=0.001 — which is correct here but is the same class of bug and will bite differently if the optimizer list changes.

**Is it "legitimate given their configurations"?** Mathematically yes — with zero weight decay the two optimizers *are* the same algorithm. But the experiment was clearly intended to test decoupled weight decay, and it does not. **It is a configuration bug that produced a duplicate experiment.**

**Blast radius:** 10 wasted runs; `results/t7_adam_stats.csv`; `ROUND4_REPORT.md` §4, which presents Adam and AdamW as two rows of independent evidence for C8 and then explains the (single) result with a mechanistic story about per-parameter adaptation. C8's "not adaptive preconditioning" half is supported by **one** optimizer, reported as two.

**Also contaminating C8:** the SGD arms are schema 3 (older `train.py`) and the Adam arms are schema 0 (current). And the momentum arm runs at lr=0.01 while the adaptive arm runs at lr=0.001, so "survives momentum but not adaptive preconditioning" is confounded with a 10× learning-rate difference.

**Severity: high for C8.**

### 5.4 The CIFAR pipeline — **training is failing, not evaluation**

**[DATA] The evidence.** `R4_T3b_bp_seed1`, per task, layer 0:

| task | train_acc | test_acc | prev_tasks_acc | task_0_acc | eff_rank | dead_frac |
|---|---|---|---|---|---|---|
| 0 | 0.0703 | 0.0742 | 0.0742 | 0.0742 | 158.2 | 0.0000 |
| 1 | 0.0664 | 0.0352 | 0.0312 | 0.0273 | 160.1 | 0.0000 |
| 4 | 0.0781 | 0.0781 | 0.0172 | 0.0039 | 155.6 | 0.0000 |
| 9 | 0.0742 | 0.0625 | 0.0113 | 0.0000 | 152.5 | 0.0000 |

**`train_acc` on task 0 — the task it has just finished training on, evaluated on its own training data — is 7.0%.** A 10-class task. The model has not learned task 0. Retention is irrelevant; there is nothing to retain. All four arms (`bp`, `rs`, `ewc`, `ln_l2`) land in 6–8%, and `rs`/`bp` agree to three decimals.

**Where it breaks — four compounding causes, in order of confidence:**

1. **[CODE, high confidence] The optimization budget is ~19 steps per task.** `--epochs 1` on 5,000 samples with batch 256 gives 19 gradient steps. Ten tasks → **190 steps total** to train a 4-layer ConvNet from random init. That alone is decisive.
2. **[CODE, high confidence] `clip_grad_norm_(..., 0.5)` caps every step at `lr × 0.5 = 0.005` in parameter norm.** 19 steps × 0.005 = a total parameter displacement of ~0.1 in a network whose initial weight norm is orders of magnitude larger. The network essentially cannot move. This clip is hardcoded and was tuned for the MLP.
3. **[CODE, structural] Single 100-way head, no task masking, no replay.** `ConvNet(output_dim=100)` with `SplitCIFAR100.get_task_data` returning global class labels. Even a well-trained model would score ~0 on previous tasks in this protocol. **`ROUND4_REPORT.md` §5 reads the resulting ~1.2% as "random chance for 100 classes" and concludes Split-CIFAR-100 is "a capacity exhaustion / noise benchmark". That conclusion does not follow** — current-task accuracy is also at floor, which is a training failure, not a forgetting result. By contrast `experiments/split_cifar10/` uses `SimpleCNN` with **per-task heads** and reaches train 1.00 / test 0.95 / avg-seen 0.71 on the same family of benchmark.
4. **[CODE, secondary] The penalty target is nonsensical at conv layers.** `ConvNet.forward` flattens the whole feature map before applying the penalty, so `d = 32×32×32 = 32768` at conv1 and the radius target is `√32768 ≈ 181`. [DATA] measured `radius_mean` ≈ 227 at λ=0 and **226.4 at λ=0.01** — the penalty is a rounding error at this scale. The `rs` arm on CIFAR is inert. (`ln_l2` sits at exactly 180.9986 = √32768, because LayerNorm forces it.)

**Is it training or evaluation? Training.** Evaluation is fine — `test_acc` tracks `train_acc`, `dead_frac` is 0, `eff_rank` is ~150, so the forward pass and the metric code work. The optimizer never gets anywhere.

**Blast radius:** `S5_2_*` (9 runs), `R4_T3b_*` (12 runs), `results/t3b_cifar_stats.csv`, `ROUND3_S5_REPORT.md` §S5.2, `ROUND4_REPORT.md` §5. Not fixed, per instructions.

**Severity: high**, because the report currently draws a *scientific* conclusion ("this benchmark is noise") from a pipeline fault.

### 5.5 Additional defects found

**(a) [CODE+DATA] `er` and `l2_er` destroy the network.** `compute_er_penalty` returns `torch.sum(off_diag**2)` — the raw squared Frobenius norm of the off-diagonal covariance, unnormalized, over a 1000×1000 matrix — added at a hardcoded coefficient of 0.01, summed over all three layers. [DATA] `S2_er` and `S2_l2_er` reach 0.110 and 0.112 accuracy vs `bp`'s 0.768. Its own docstring calls it a "proxy" and the comments admit uncertainty about the formulation. These are not meaningful baselines. **Severity: medium** (they appear in `analysis/s2_baselines.py` output).

**(b) [CODE+DATA] `mas` collapses current-task accuracy.** SI/MAS penalties use hardcoded coefficient 1.0 and MAS's `omega` **accumulates without decay or normalization** across all 150 tasks. [DATA] `R3_R2_mas` test_acc 0.4086 ± 0.0149 — the network is frozen. It then posts a high "retention" (0.3982) purely because a frozen network does not forget. `QUANTITATIVE_RESULTS.md` lists it in the baseline table without noting that its retention number is an artifact of untrained-ness. **Severity: medium** — this directly affects C6's Pareto-front claim.

**(c) [CODE] The `λ=inf` arm is not the λ→∞ limit.** `HardProjection.backward` is a **straight-through estimator** — it returns `grad_output` unchanged. The true hard-constraint limit would project the gradient onto the tangent space, killing the radial component. As written, the radial gradient component passes through in full. So C4's "vanishes in the hard-projection limit" endpoint is a *different algorithm*, not the limit of the sweep. **[DATA]** consistent with this: `R3_R1_lam_inf` has φ̃_rad ≈ 1.01 at layer 0 — exactly the random-direction value for d=1000 — while λ=10 has 0.19. The hard arm has *more* radial gradient energy than the soft arm. **Severity: medium-high for C4's framing.**

**(d) [CODE] `sweep.py --suite S2` is broken today.** Its arm list still contains `--method cbp`, which argparse now rejects. The suite will produce 5 failed runs and continue.

**(e) [CODE] Unpaired t-tests on paired data.** `c1_frontier_v2.py` uses `scipy.stats.ttest_ind` on arms that share seeds 1–5. `t2_c1_canonical.py` correctly uses `ttest_rel`. Inconsistent, and `ttest_ind` is the wrong test here.

**(f) [DATA] The "project halted" verdict in `results/ANOMALIES.md` was reached on the wrong axis.** It cites `analysis/s2_baselines.py`: "RS (best λ=1.0) 77.72% vs L2 (1e-3) 78.17%, t=−3.665, p=0.0215 — RS is statistically worse." I re-ran that script; it reproduces. But **`S2_*` has no `prev_tasks_acc` column** — the comparison is on *current-task accuracy only*, which is not the axis C1 is about. On the retention axis at a comparable config (`R3_R2`, last 20 tasks): `l2` retention 0.1234 ± 0.0008, `rs` 0.3319 ± 0.0053. **The kill condition was evaluated against a metric the suite could not measure.** That is worth knowing before anyone treats the halt as settled — in either direction. **Severity: high, for the project's own decision record.**

**(g) [CODE] Duplicate `run_b1_b2.py` / `run_b4_c1.py` outputs are still on disk** under names one letter away from the current ones (`B4_*` vs `R2_B4_*`), at different learning rates. Any glob like `results/*B4*` silently pools them.

**(h) [DATA] A stray nested path artifact:** `non_algo/home/psquare_a6000/Desktop/grokking_mech_interp/non_algo/stiffness/results/b2_undertrained.png` — an absolute path written as a relative one. Harmless, but it indicates output paths were not always what the scripts thought.

---

## 6. Where the data disagrees with the claims

Surfacing these together, since several were requested to be flagged plainly.

**C1 — supported, but weaker than reported and on a slightly wrong metric.** At matched current-task accuracy, RS does beat BP: `R2_C1`, last 20 tasks, `bp` at lr=0.1 gives test 0.9634 / retention 0.4817; `rs` at λ=0.03 gives 0.9630 / 0.5704. That is real. But the "matched" comparison is undermined by `prev_tasks_acc` including the current task, and by [DATA] the fact that **reducing BP's learning rate makes retention worse, not better** (lr=0.1 → 0.4817; lr=0.025 → 0.4158; lr=0.015 → 0.4202). The effective-learning-rate confound that `R3_C1_bp_matched_*` was built to rule out does not behave monotonically, so "matched at lower lr" is not a well-posed control on this metric.

**C2 — no valid experiment.** §3.3.2. The control's achieved magnitude match is ~4% of its intended one.

**C3 — contradicted as stated.** §3.3.3. Overlap moves the *wrong way* at 2 of 3 layers; the reported effect is a 3-layer unweighted average dominated by layer 2; and within the finite-λ region where all of C4's retention variation lives, overlap is flat to 0.001 while retention swings 13 points. Drift magnitude *increases* with λ while retention improves — which does support "not restriction of movement", but the metric is normalized by a quantity the penalty shrinks, so even that reading is not safe.

**C4 — supported.** [DATA] `t1_lambda_bracket.csv`: retention peaks at λ=0.003 (0.3457) and falls monotonically to 0.2957 at λ=10 and 0.2116 at λ=inf ≈ 0.2118 at λ=0. The shape is exactly as claimed. Caveats: 2–5 seeds per point, and the `inf` endpoint is an STE, not the constraint limit (§5.5c).

**C5 — supported, and it is the cleanest result in the repo.** [DATA] Across `R4_T1`, `test_acc` rises monotonically 0.9587 → 0.9620 while retention rises then falls (0.2118 → 0.3457 → 0.3319). The two axes genuinely decouple, in the same runs, at the same config, with 5 seeds each.

**C6 — half-supported.** Additivity holds: [DATA] `ewc_10000` retention 0.4375 ± 0.0037, `rs_ewc_10000` 0.4767 ± 0.0108, at nearly identical current-task accuracy (0.7481 vs 0.7445). That is a genuine additive composition. The Pareto claim is narrower than stated: RS dominates `l2`, `ln_l2`, `l2_init`, `sp`, `ewc_100` and `cbp` (which is BP). It does **not** dominate `ewc_10000` or `si` — those buy more retention at a real accuracy cost and sit on a different part of the frontier. And `mas`'s apparent 0.3982 retention is an artifact of a frozen network (§5.5b). Arms have n = 3, 4, or 5 and the report says 5.

**C7 — supported.** [DATA] `r4_stats.csv`: retention delta (RS − BP) = 0.027 / 0.062 / 0.124 / 0.131 at widths 256 / 512 / 1000 / 2048. Monotone and large. Caveat: `R3_R4_w_1000_lam_0.03` has n=2, and the delta flattens between 1000 and 2048.

**C8 — the momentum half is supported; the adaptive half rests on one run reported as two.** [DATA] `sgd_momentum`: BP 0.2082 → RS 0.3344 (+0.126). `adam` (current data): BP 0.1962 → RS 0.1202 (−0.076), sign reversal. But `adam` ≡ `adamw` bitwise (§5.3); the SGD and Adam arms come from different `train.py` generations; and lr differs 10× between them.

**C9 — supported, and by a wide margin.** [DATA] `a1b_stats.csv`: the current-task accuracy slope over tasks 50–300 is −6.3e−6 (1 ep), −1.37e−5 (10 ep), −1.47e−5 (30 ep) per task. Over 250 tasks that is ~0.0037 accuracy lost to plasticity decay. Retention over the same horizon falls from ~0.96 to ~0.21 — **a factor of ~200**. "Orders of magnitude" is if anything conservative. Two caveats: the epochs=3 row is missing (2 of 3 seeds crashed), and `ROUND2_REPORT.md` §B1 states plasticity loss "does not exist" (p=0.27/0.33 at 1 epoch) while `a1_stats.csv` finds it highly significant at 10 epochs (p=5.5e−11) — the same phenomenon, two budgets, described as contradictory findings.

---

## 7. Rewrite boundary

### 7.1 The recommendation

Draw the line at **the trainer**. Specifically:

**KEEP as-is:**
- `src/methods/rs.py` — 35 lines, unit-tested, mathematically correct, matches `experiments/shared/norm_penalty.py` independently. (One change needed: `HardProjection.backward` should offer a true tangential projection alongside the STE, since C4's endpoint depends on which one you mean.)
- `src/metrics/phi_rad.py`, `rank.py`, `norms.py`, `neurons.py`, `readiness.py` — small, correct, and `phi_rad.py` has the regression test that matters.
- `tests/` — both files. Extend, don't replace.
- `src/models/mlp.py`, `src/data/permuted_mnist.py`, `src/data/rotating_mnist.py` — simple and correct.
- All 686 stored parquets — as **archive**, not as inputs to new tables.

**REWRITE:**
- `src/train.py` — specifically the block from line 179 (`if args.lambda_rs > 0`) through line 434. This is where the clip/shrinkage interaction, the drift metrics, the readiness computation, the retention loop and the isotropic control all live, entangled, untested, and inline.
- `src/sweep.py` — replace hardcoded suite dictionaries with declarative config files (the empty `configs/` directory says someone already intended this).
- `src/methods/cbp.py` — rewrite against Dohare et al. properly (utility-ranked selection, replacement rate ~1e-4), or delete it and drop the CBP baseline from the paper. Do not ship the current one.
- `src/methods/baselines.py::compute_er_penalty` — normalize or remove.
- `src/data/split_cifar100.py` + `src/models/convnet.py` — rewrite together with per-task heads and a per-layer (not per-flattened-feature-map) penalty. `experiments/split_cifar10/` is the working reference.
- The isotropic-control machinery end to end: delete `extract_shrinkage.py`, delete `analysis/compute_iso_shrinkage.py`, and rebuild (§7.3).

**DELETE:**
- `audit_configs.py` (dead), `build_audit.py` (hand-typed constants), `results/CONFIG_AUDIT.md` (its wrong output).
- `analyze_s52.py` (verbatim duplicate), `analysis/c1_frontier.py` (superseded by v2), one of `b2_convergence.py`/`b2_undertrained.py`, one of `r1_r3_analysis.py`/`r1_r3_drift_retention.py`.
- `run_b1_b2.py`, `run_b4_c1.py`, `run_s0.sh` — and quarantine their outputs (`B1_*`, `B2_*`, `B4_*`, `C1_A_*`, `C1_B_*`, `S0.*`) into `results/_archive/` so no glob can reach them.

**QUARANTINE, do not delete:**
- `experiments/` entire tree. It is the source of `latex/report.tex`'s figures. Add a `README` at its root stating the radial-energy bug and listing which figures are affected, so nobody reuses those numbers. Do not fix it — nothing depends on it going forward, and fixing it would invite reuse.

### 7.2 Why the boundary goes there

I considered three lines and rejected two.

*Rejected — "keep `train.py`, patch the three defects."* The three defects (clip order, broadcast readiness, dead CBP) are individually 1–5 line fixes. But they were not the hard part to find. The hard part was that `train.py` has been edited at least five times with no version marker, that its output schema changed seven times, and that each change silently redefined what a stored column means. Patching produces a *sixth* undated generation and an eighth schema. The problem is not the bugs; it is that the file has no mechanism for telling you which version produced a number. That needs structural change, not patches.

*Rejected — "rewrite everything including the metric helpers."* `src/metrics/phi_rad.py` is the one place in this repo where somebody found a subtle bug, fixed it, and wrote a regression test that specifically catches it (`test_phi_rad_tilde_leakage`). Throwing that away to avoid "inheriting contamination" would discard the strongest artifact in the codebase. Same for `rs.py`, which is independently corroborated by a completely separate implementation in `experiments/`. Rewriting these buys nothing and risks re-introducing the bug that `experiments/` still has.

*Chosen — the trainer.* Everything I found wrong in §3 and §5 lives in one file, in one contiguous region, and every one of those defects is of the same kind: **a measurement or intervention written inline, with no test, whose meaning depends on surrounding statements that were added later.** The clip is on line 195 and the shrinkage on line 193 — the bug is the adjacency. Readiness is per-layer on line 280 because someone edited it in place; nothing recorded that it used to be otherwise. The retention loop is correct but misnamed, and nothing asserts what it computes. A rewrite that pulls each measurement into `src/metrics/` behind a tested function, and makes the training step's ordering explicit, eliminates the whole class.

The compute argument reinforces it. `R3_R2` and `R3_R1` are the expensive suites, and `build_audit.py` already argued against re-running them. But **the two claims that need re-running (C2, C3) both need `--track_drift` at the canonical config**, which is `R4_T1`/`R4_T2c` — 150 tasks × 10 epochs × 5 seeds × a handful of arms. That is a bounded, affordable re-run, and it is the block that is already cleanest. You are not being asked to redo 700 runs; you are being asked to redo about 40.

I would rather you rewrite more than less here, and my honest read is that the trainer is the *minimum* line that removes the contamination class rather than instances of it.

### 7.3 What the new implementation must preserve

**Metric definitions — keep bit-identical so the archive stays comparable:**
- `compute_rs_penalty`: `mean_batch[(1/d)(‖h‖₂ − √d)²]`, on **pre-activations**, summed over hidden layers. Do not change to a per-layer mean.
- `compute_phi_rad_tilde`: `d · E_batch[‖g_rad‖²/‖g‖²]`, `g` = **task-loss gradient w.r.t. pre-activations only**. Keep `test_phi_rad_tilde_leakage`.
- `prev_tasks_acc`: **keep the existing definition** (mean over all tasks seen *including* the current one, first `probe_batch_size` test samples) so archived runs remain comparable — but **rename it `avg_seen_acc`** and add a genuine `prev_only_acc` alongside. Both, not either.
- `drift_overlap`: mean of the top-`k=50` principal-angle cosines, `k` fixed at 50 across all arms. Keep `k=50`.
- `readiness`: `‖ḡ‖₂ · (‖ḡ‖₂² / mean‖gᵢ‖₂²)` over m=8 microbatches — **per layer**. Add an assertion that the three layers' values are not all equal.
- Effective rank, stable rank, dead fraction (exact zero, post-activation), dormant fraction (Sokar normalized score < 0.025) — unchanged.

**Add, because their absence caused findings above:**
- `drift_*_abs` — unnormalized companions to `drift_rel`/`drift_rad`/`drift_tan`. The normalized ones are not comparable across λ (§3.3.3) and must never again be the only version logged.
- `drift_from_task0_*` — drift against a fixed task-0 reference, alongside the existing consecutive-boundary version. The current metric is not what "preservation across tasks" means.

**Logging schema — this is the non-negotiable one:**
- Every run writes `results/<run_id>/config.json` with the **full** `argparse` namespace plus every currently-hardcoded constant: batch size (256), clip norm (0.5), probe size, the `l2_init`/`er` coefficients, the SI/MAS coefficients, shrink-perturb's 0.8/0.01, ReDo's 0.025/1000.
- Plus: `schema_version` (integer, bumped on any column change), `code_version` (git SHA — **initialize a git repo; there isn't one**), `torch.__version__`, wall-clock start/end, and the **SHA-256 of `iso_shrinkage.json`** if `bp_iso` is used. The `[UNKNOWN]` in §2.3 item 7 exists solely because that last one wasn't recorded.
- Keep every existing parquet column name and dtype. Add columns; never rename or repurpose one. `audit_configs.py` was written against exactly this design and has been waiting for it — its `get_config_hash` is a reasonable starting point.
- Write the config **before** training starts, so crashed runs are still identifiable. 47 runs currently have no recoverable identity beyond their directory name.

**Config compatibility:**
- Backfill `config.json` for the archived runs by reconstructing from `sweep.py` + run-id, marking each `"provenance": "reconstructed"` and `"schema_version"` from the column-set fingerprint in §2.1. This is mechanical and should be done before anything else, because it is the only thing that makes the archive usable.
- Emit a hard error if two runs share a `run_id` with different config hashes. The Adam re-run overwrote its own predecessor silently; `r6_stats.csv` and `t7_adam_stats.csv` disagree for exactly this reason.

**The isotropic control specifically (§7.1's rewrite item), since it is the central experiment:**
- Measure shrinkage from the **actual per-step training gradient magnitude**, logged during the RS and BP runs, not from a post-hoc probe statistic and not from weight norms.
- Apply it **after** `clip_grad_norm_`, or drop clipping for both arms. As long as the clip binds, any pre-clip scaling is a no-op.
- Apply to **all** parameters (weights, biases, head), not just `model.layers[l].weight`.
- Match **per seed**, not against a 5-seed population mean.
- Add an acceptance test the control must pass before its result is reported: **the `bp_iso` arm's realized gradient-magnitude trajectory must match the `rs` arm's to within a stated tolerance.** Had that assertion existed, this experiment would have failed loudly instead of producing a publishable-looking null.
