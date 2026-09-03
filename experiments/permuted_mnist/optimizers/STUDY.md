# S7_optimizers

**Question.** Does the effect survive momentum, and does it survive adaptive preconditioning?

**Supports.** Supporting. Single code generation; lr stated per optimizer.

**Status.** complete — 18 complete, 0 failed, 0 not started, of 18 planned runs.

## Finding

<!-- FINDING -->
The effect **survives momentum and reverses under Adam.**

Retention (`prev_only_acc`, final-20-task window, paired by seed, n=3 per arm), at
current-task accuracy matched to within 0.005 across every arm:

| optimizer | lr | baseline | penalty (λ*=0.003) | delta | t | p | wins |
|---|---|---|---|---|---|---|---|
| SGD | 0.1 | 0.2020 | 0.3433 | **+0.1412 ± 0.0026** | +92.58 | 1.2e-04 | 3/3 |
| SGD + momentum | 0.01 | 0.1990 | 0.3460 | **+0.1469 ± 0.0060** | +42.39 | 5.6e-04 | 3/3 |
| Adam | 0.001 | 0.1758 | 0.1196 | **−0.0562 ± 0.0192** | −5.07 | 3.7e-02 | 0/3 |

`test_acc` is 0.9554–0.9600 across all six arms, so these are retention differences at matched
current-task accuracy, not a plasticity trade.

The SGD result reproduces the stiffness curve (S3 gives 0.207 → 0.342 at λ*). Momentum leaves
it intact — slightly larger, well within seed noise. **Under Adam the penalty does not merely
weaken, it is harmful**: retention drops below the unpenalised baseline, with a clean 0/3 sign
split.

This is the prediction the paper's decomposition makes, and stronger than stated. Momentum
rescales the *update history* but not the geometry of activation space, so the radial/tangential
split is preserved; Adam rescales each coordinate independently by its own gradient statistics,
which does not preserve that split. The sign reversal says the penalty under a diagonal
preconditioner is not a weaker version of the same mechanism but a different intervention.

Supporting evidence in the norms: `adam/baseline` reaches weight_norm 478 and radius 515,
against ~50 for both SGD arms. Adam is already operating in a completely different norm regime,
and the penalty pulls radius to 143 — a large intervention that still costs retention.

**Caveat that must be stated.** λ is held at 0.003 for all three optimizers, and that value was
tuned for SGD at lr 0.1 (S3). A weakened or reversed effect under Adam is therefore consistent
with two readings: the mechanism is disrupted by diagonal preconditioning, or λ* is simply
mis-tuned for Adam's effective step size. Separating them needs a per-optimizer stiffness curve
(~8 λ × 3 seeds per optimizer), which this study does not run.

Per-optimizer learning rates are stated in the table above and are not a footnote: sgd 0.1,
sgd_momentum 0.01, adam 0.001. No two arms agree to four decimals (checked); no config_hash
collisions.

18/18 runs complete, 0 failed. Single code generation, git_sha bff5bd9.
<!-- /FINDING -->

## Configuration

Shared by every arm unless the arm table says otherwise. Read from the study registry (`src/studies.py`); each run's actual configuration is in its own `config.json`.

| key | value |
|---|---|
| `act_fn` | `relu` |
| `batch_size` | `256` |
| `clip_norm` | `0.5` |
| `dataset` | `permuted_mnist` |
| `depth` | `3` |
| `epochs` | `10` |
| `lr` | `0.1` |
| `n_tasks` | `150` |
| `optimizer` | `sgd` |
| `probe_size` | `2000` |
| `spectral_every` | `5` |
| `width` | `1000` |

Seeds: `[1, 2, 3]`

## Arms

| arm | seeds complete | status |
|---|---|---|
| `adam/baseline` | 3/3 | complete |
| `adam/penalty` | 3/3 | complete |
| `sgd/baseline` | 3/3 | complete |
| `sgd/penalty` | 3/3 | complete |
| `sgd_momentum/baseline` | 3/3 | complete |
| `sgd_momentum/penalty` | 3/3 | complete |

## Phases

0. `sweep` � 6 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S7_optimizers --tmux
```
