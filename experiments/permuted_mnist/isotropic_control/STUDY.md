# S1_isotropic_control

**Question.** Is the penalty's retention benefit reproduced by an equal-magnitude, direction-unconstrained intervention?

**Supports.** Paper claim 3 (mechanism: direction vs magnitude). Decisive.

**Status.** planned — 0 complete, 0 failed, 40 not started, of 40 planned runs.

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
| `log_grad_trace` | `True` |
| `lr` | `0.1` |
| `n_tasks` | `150` |
| `optimizer` | `sgd` |
| `probe_size` | `2000` |
| `track_drift` | `True` |
| `width` | `1000` |

Seeds: `[1, 2, 3, 4, 5]`

## Arms

| arm | seeds complete | status |
|---|---|---|
| `clipped/arm_baseline` | 0/5 | 0/5 done |
| `clipped/arm_isotropic_global` | 0/5 | 0/5 done |
| `clipped/arm_isotropic_per_layer` | 0/5 | 0/5 done |
| `clipped/arm_penalty_lam0.003` | 0/5 | 0/5 done |
| `unclipped/arm_baseline` | 0/5 | 0/5 done |
| `unclipped/arm_isotropic_global` | 0/5 | 0/5 done |
| `unclipped/arm_isotropic_per_layer` | 0/5 | 0/5 done |
| `unclipped/arm_penalty_lam0.003` | 0/5 | 0/5 done |

## Phases

0. `clipped_A_targets` — 2 arm(s)
1. `clipped_B_control` — 2 arm(s)
2. `unclipped_A_targets` — 2 arm(s)
3. `unclipped_B_control` — 2 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S1_isotropic_control --tmux
```
