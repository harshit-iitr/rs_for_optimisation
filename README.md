# Radial suppression as an optimization intervention

A soft penalty pulls hidden pre-activations toward a fixed-radius sphere:

```
L_pen(h) = (1/d) * (||h||_2 - sqrt(d))^2
```

We study what it does to optimization dynamics, using a long task sequence as the
measurement instrument. Paper draft in [`latex/`](latex/) (OPT 2026).

## Layout

```
src/            trainer, launcher, study registry, metrics, methods
tests/          regression tests -- run these before trusting a change
analysis/       analysis scripts; analysis/common.py enforces the reporting rules
experiments/    one directory per study; see experiments/README.md
  _archive/     Rounds 1-4, frozen. Read its README before using anything there.
latex/          the paper
opt2026_style/  pristine OPT 2026 template, for diffing against latex/
```

## Getting started

```bash
pip install -r requirements.txt
PYTHONPATH=. python3 -m pytest tests/ -q          # 30 tests, all should pass
PYTHONPATH=. python3 -m src.launch --study S3_stiffness_curve --dry-run
```

MNIST downloads on first run into `data/`, which is gitignored.

## Adding an experiment

Studies are **data**, not code. Add an entry to `STUDIES` in
[`src/studies.py`](src/studies.py) giving the question it answers, which claim it
supports, its seeds, and its arms. The launcher materialises it into
`experiments/<benchmark>/<study>/<arm>/seed_N/`, so a run's path states what it
is and no run-id decoding is needed.

```bash
PYTHONPATH=. python3 -m src.launch --study MY_STUDY --dry-run
PYTHONPATH=. python3 -m src.launch --study MY_STUDY --gpus 0 --concurrency 8 --threads 16 --tmux
PYTHONPATH=. python3 -m src.docs                  # regenerate README + STUDY.md from run status
```

The launcher is resumable, gates concurrency on measured free GPU memory, retries
runs that die, and records failures in `_launch_report.json`. A run completed
under a configuration that no longer matches the study is treated as stale and
re-run, so a study cannot silently mix configurations.

## Core metrics

Definitions are given here because several of them are easy to misread, and one
of them was misread for four rounds. `h` is a layer's pre-activation matrix over
the probe batch, shape `(B, d)`. Everything below is computed **per layer** and
is **never averaged across layers**. Implementations are in `src/metrics/`.

### Accuracy and retention

Let `a_t^(k)` be accuracy on task `k`'s held-out set after finishing task `t`,
measured on the first `probe_size` test examples of that task.

| column | definition | note |
|---|---|---|
| `test_acc` | `a_t^(t)` | current-task accuracy |
| `prev_only_acc` | `mean over k < t of a_t^(k)` | **the retention metric.** NaN at `t = 0` |
| `avg_seen_acc` | `mean over k <= t of a_t^(k)` | includes the current task |
| `task_0_acc` | `a_t^(0)` | accuracy on the first task only |

`avg_seen_acc` is what Rounds 1–4 called "previous-task accuracy". It **includes
the current task**, so it is inflated by `test_acc / (t+1)` and is biased toward
arms with higher current-task accuracy, which is exactly the confound that "at
matched current-task accuracy" is trying to remove. It is kept only so archived
runs stay comparable. **Report `prev_only_acc`.**

### Radial geometry

With `r = ||h||_2` per example, `d` the layer width:

| column | definition |
|---|---|
| `radius_mean` | mean of `r` over the probe batch |
| `radius_std` | std of `r` over the probe batch. Exactly `0` under hard projection |
| `radial_excess` | `radius_mean - sqrt(d)`. Zero when the constraint is exactly satisfied |

### Radial gradient energy (`phi_rad_tilde`)

With `h_hat = h / ||h||_2` and `g` the gradient of the **task loss only** with
respect to pre-activations,

```
g_rad     = (g . h_hat) h_hat
phi_rad   = ||g_rad||^2 / ||g||^2          per example, then averaged
phi_rad_tilde = d * phi_rad
```

The `d` factor normalises against the isotropic null: for `g` in a uniformly
random direction the radial subspace is one-dimensional out of `d`, so
`E[phi_rad_tilde] = 1`. Values below 1 mean the task gradient is more tangential
than chance.

**`g` must be the task-loss gradient, never the total gradient.** The penalty's
gradient is purely radial by construction, so including it makes the metric
partly a measurement of the regularizer instead of the task. This is exactly the
bug that invalidated the Round 1–4 diagnostic; `tests/test_phi_rad.py::test_phi_rad_tilde_leakage`
exists to catch it. The trainer obtains `g` by `torch.autograd.grad` of a probe
loss that never contains the penalty.

Two companion columns let you decompose a change in `phi_rad_tilde` into
numerator and denominator, which matters because the two tell different stories:
`g_rad_norm` is the mean per-example `||g_rad||`, and `g_norm_task` is the mean
per-example `||g||`.

### Drift

Measured on a probe of `drift_probe_size` task-0 inputs, held fixed for the whole
run. Given current and reference representations,

```
delta      = h_curr - h_ref
delta_rad  = component of delta along h_ref, per example
delta_tan  = delta - delta_rad
```

| column | definition |
|---|---|
| `drift_abs` | `||delta||_F` |
| `drift_rad_abs`, `drift_tan_abs` | Frobenius norms of the two components |
| `drift_rel`, `drift_rad`, `drift_tan` | the same three divided by `||h_ref||_F` |
| `drift_cos_sim` | mean per-example cosine similarity |
| `*_ref` suffix | same quantities against a **fixed task-0 reference** rather than the previous task boundary |

Without the `_ref` suffix the reference is the **previous task boundary**, so it
is a per-task increment, not cumulative movement.

**Use the absolute forms for comparisons across penalty strengths.** The relative
forms divide by `||h_ref||`, which the penalty directly shrinks, so an identical
physical displacement reads larger in a penalised arm. The relative forms exist
for comparability with archived runs.

### Rank, units, readiness

| column | definition |
|---|---|
| `eff_rank` | `exp(-sum_i p_i log p_i)` with `p_i = sigma_i / sum_j sigma_j` |
| `stable_rank` | `||A||_F^2 / ||A||_2^2` |
| `subspace_overlap` | mean cosine of principal angles between the top-`k` left singular subspaces, `k = 50` fixed for every arm |
| `subspace_proj_metric` | mean **squared** cosine of the same angles |
| `dead_frac` | fraction of units whose post-activation is exactly zero on every probe input |
| `dormant_frac` | fraction whose mean absolute activation is below `0.025` of the layer mean |
| `readiness` | over `m = 8` probe microbatches, `\|g_bar\| * \|g_bar\|^2 / mean_i \|g_i\|^2`, computed **per layer** |
| `weight_norm`, `grad_norm` | Frobenius norms of the layer weight and its probe gradient |
| `update_norm` | `||W_after - W_before||_F` on the last optimizer step of the task |

`k` is held at 50 regardless of width so the same rank of subspace is compared
across arms. All spectral quantities are computed on CPU in float64: the GPU path
raises on ill-conditioned activation matrices and its internal fallback is
roughly 400x slower. A failure returns NaN and increments a counter recorded in
the run's `config.json`, so a run with degraded diagnostics is identifiable
rather than silently wrong.

## Conventions that are not optional

These exist because Rounds 1-4 violated each of them and produced results that
had to be withdrawn. The audit is at
`experiments/_archive/round1_4_stiffness/results/REPO_AUDIT.md`.

1. **Every run writes `config.json` before training starts**, including every
   constant that would otherwise be implicit, a schema version and the code
   revision. Nothing is reconstructed from a directory name.
2. **An arm with missing seeds is not reported.** `analysis/common.py` raises
   rather than averaging the survivors. Pass `--preliminary` to override, and
   every row is then stamped with its true `n`.
3. **Per-layer metrics stay per layer.** Never averaged across layers.
4. **Comparisons report test statistic, p, n and sign split**, paired where seeds
   align.
5. **Identical values across arms to four decimals are a bug signal.** Two
   optimizer arms once matched bitwise because both had weight decay zero.
6. **An arm that fails its own acceptance test is not reported at all.**
7. **Task-loss gradient only in `phi_rad_tilde`.** There is a regression test.
8. **Anything that rescales gradients runs after `clip_grad_norm_`, never
   before.** Applied before, the clip renormalises it away and the intervention
   silently does nothing. `tests/test_isotropic.py::test_clip_after_scale_destroys_the_match_THE_ROUND_1_4_BUG`
   is named after the bug this caused.

## Regenerating figures and analyses

```bash
PYTHONPATH=. python3 analysis/figures.py                       # paper figures from run data
PYTHONPATH=. python3 analysis/s1_isotropic.py --preliminary
PYTHONPATH=. python3 analysis/s2_s3_frontier.py --preliminary
PYTHONPATH=. python3 analysis/characterization.py              # full metric battery
```

## Not in this repository

- `data/` -- MNIST and CIFAR, re-downloadable.
- `*.pt` checkpoints.
- `grad_trace.npz` -- per-step gradient and weight-norm traces, 335 MB. Needed
  only to build the isotropic control arms; regenerate with `--log_grad_trace`.
  The launcher runs the target arms before any arm that consumes them.

---

# Task allocation

Six tasks, two each. Tasks 5-7 need **no new code**: the studies are already
defined in `src/studies.py` and every method they need is already in the trainer.
Tasks 9-11 need code and are described with what to build.

Before starting anything, read [Core metrics](#core-metrics) and
[Conventions](#conventions-that-are-not-optional) above. If you are using a
coding agent, give it this README.

## Harshit — baselines, and curvature

### T5. Baselines (`S6_baselines`, 39 runs, no new code)

The paper argues the penalty acts as an anisotropic weight regularizer whose axes
rotate with the task, and predicts that an isotropic weight penalty of matched
strength cannot reproduce the benefit. There is currently no data for this. It is
the only claim in the paper with none.

```bash
PYTHONPATH=. python3 -m src.launch --study S6_baselines --dry-run
PYTHONPATH=. python3 -m src.launch --study S6_baselines --gpus 0 --concurrency 8 --threads 16 --tmux
PYTHONPATH=. python3 analysis/sweeps.py S6_baselines
```

Arms: baseline, weight decay at two strengths, LayerNorm+WD, regularize-to-init,
shrink-and-perturb, EWC at three strengths, SI, MAS, the penalty, and
penalty+EWC. Report the frontier scatter, not a single column: the claim is
Pareto position, not dominance. Note in `STUDY.md` that MAS reaches very low
current-task accuracy, so its retention is an artifact of a network that never
trained, and say so rather than letting it read as a strong baseline.

### T9. Curvature diagnostics (~40 lines)

Adds the missing row of the diagnostic battery.

- New `src/metrics/curvature.py`: Hessian trace by Hutchinson with Rademacher
  probes, and top singular value by power iteration. Report the probe count and
  show the estimate has converged.
- Call it at the task-boundary probe in `src/train.py`, add `hessian_trace` and
  `sigma_max` to the logged row, bump `SCHEMA_VERSION` in `src/version.py`, add a
  test.
- Then re-run the arms you want the column for.

**Trap:** bumping `SCHEMA_VERSION` does *not* trigger re-runs, because the resume
check compares study arguments and not the schema version. Existing runs will
carry NaN in your new columns. Re-run the arms you need deliberately, and say in
`STUDY.md` which arms have the column and which do not.

## Aditya — optimizers, and the selectivity figure

### T6. Optimizer interaction (`S7_optimizers`, 18 runs, no new code)

This is the highest-value task for acceptance. OPT 2026's stated theme is
frontier optimizers, and the paper currently contains no optimizer comparison.
The decomposition in the paper is stated for gradient flow; a diagonal
preconditioner rescales coordinates independently and does not preserve the
radial/tangential split in activation space, so the effect should survive
momentum and weaken under Adam.

```bash
PYTHONPATH=. python3 -m src.launch --study S7_optimizers --gpus 0 --concurrency 8 --threads 16 --tmux
PYTHONPATH=. python3 analysis/sweeps.py S7_optimizers
```

Arms: SGD, SGD+momentum, Adam, each with and without the penalty, at a learning
rate chosen per optimizer and **stated in the table**. Rounds 1-4 compared
optimizer arms across two code generations at a 10x learning-rate difference, and
its two adaptive arms turned out to be bitwise identical because both had weight
decay zero, so `AdamW(0) == Adam(0)`. Check for that: if two arms agree to four
decimals, find out why before reporting.

### T10. Per-neuron task selectivity (~80 lines)

The paper's most memorable figure, and it does not exist yet.

- Add `--save_final_checkpoint` to `src/train.py`.
- New `analysis/selectivity.py`: load an end-of-run checkpoint, and for each
  hidden unit and each task compute its mean absolute response on that task's
  data, normalised per unit. The question is whether units are task-shared or
  task-specific.
- Figure in `analysis/figures.py`: units x tasks heatmap, one panel each for
  baseline, penalty, and hard projection.
- Needs ~12 arms re-run with checkpointing on. No checkpoints exist in the
  repository, so you are generating them from scratch; that is expected.

Follow the figure conventions already in `analysis/figures.py`: categorical
colours in fixed slot order, no dual axes, text in ink colours rather than series
colours, and look at the rendered output before calling it done.

## Sarthak — second benchmark, and the causal arm

### T7. Second task family (`S8_rotating_mnist`, 18 runs, no new code)

The paper is one task family, one architecture. This is the first thing a
reviewer will raise.

```bash
PYTHONPATH=. python3 -m src.launch --study S8_rotating_mnist --gpus 0 --concurrency 8 --threads 16 --tmux
PYTHONPATH=. python3 analysis/sweeps.py S8_rotating_mnist
```

Rotating MNIST is a continuous input transformation rather than a permutation, so
it is a real test of the account and not only a generality box-tick: the
anisotropy argument makes a *different* prediction when the input covariance
rotates smoothly. Report whether `phi_rad_tilde` at layer 0 still tracks
retention there. Round 1-4 tested only penalty strengths far past the optimum on
this benchmark, so use the sweep as defined.

### T11. The causal arm (~60 lines, design first)

The paper's mechanism claim is currently correlational: `phi_rad_tilde` at layer
0 tracks retention across the sweep at Spearman rho = -0.905, but the sweep is
one-dimensional in lambda, and eight aggregate points cannot separate
`phi_rad_tilde` from lambda itself. A reviewer will say so in one line.

The experiment that separates them is an arm that **lowers `phi_rad_tilde`
without constraining the radius**. If retention follows, the mechanism claim
becomes causal.

Sketch the formulation before writing code, and discuss it, because the design is
the hard part rather than the implementation. One candidate: penalise the radial
component of the task gradient directly, which needs a second backward pass and
is not simply another activation penalty. Whatever you choose, it must leave
`radius_mean` close to the unpenalised value, otherwise it is not separating the
two variables.

Add it as a new method in `src/train.py::METHODS` and a new study in
`src/studies.py`. Verify on a short run that `radius_mean` is unchanged and
`phi_rad_tilde` has moved, before spending a full sweep.

## Running at the same time

If more than one of you runs on the same machine, agree on `--gpus` and keep the
total of `concurrency x threads` under `nproc`. The launcher gates on measured
free GPU memory but has no visibility into each other's CPU threads, and the
metric SVDs are CPU-bound.

## One thing missing from the clone

`grad_trace.npz` files are not in the repository (335 MB). They are per-step
gradient and weight-norm traces, needed only by the `S1_isotropic_control` arms.
On a fresh clone `analysis/s1_isotropic.py` will report every acceptance test as
`PASSED=False (n=0 seeds)`. **That is missing data, not a failed control.**
Regenerate by re-running S1's target arms first; the launcher's phase ordering
does this automatically. None of the six tasks above touch S1.
