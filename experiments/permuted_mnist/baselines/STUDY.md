# S6_baselines

**Question.** Where does the penalty sit on the plasticity-stability frontier relative to parameter-regularization baselines, and does it compose?

**Supports.** Supporting (Pareto position and additivity).

**Status.** planned — 0 complete, 0 failed, 39 not started, of 39 planned runs.

## Finding

<!-- FINDING -->
_Not yet run._
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
| `baseline` | 0/3 | 0/3 done |
| `ewc_100` | 0/3 | 0/3 done |
| `ewc_1000` | 0/3 | 0/3 done |
| `ewc_10000` | 0/3 | 0/3 done |
| `l2_init` | 0/3 | 0/3 done |
| `layernorm_wd_1e-4` | 0/3 | 0/3 done |
| `mas` | 0/3 | 0/3 done |
| `penalty` | 0/3 | 0/3 done |
| `penalty_plus_ewc_10000` | 0/3 | 0/3 done |
| `shrink_perturb` | 0/3 | 0/3 done |
| `si` | 0/3 | 0/3 done |
| `weight_decay_1e-3` | 0/3 | 0/3 done |
| `weight_decay_1e-4` | 0/3 | 0/3 done |

## Phases

0. `sweep` — 13 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S6_baselines --tmux
```
