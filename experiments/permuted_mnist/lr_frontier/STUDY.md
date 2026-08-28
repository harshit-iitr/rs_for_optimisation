# S2_lr_frontier

**Question.** Does lowering the step size buy retention, and is the relationship monotone?

**Supports.** Paper claim 1. Becomes the reference frontier for claim 2.

**Status.** planned — 0 complete, 0 failed, 35 not started, of 35 planned runs.

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
| `lr` | `0.003` |
| `n_tasks` | `150` |
| `optimizer` | `sgd` |
| `probe_size` | `2000` |
| `track_drift` | `True` |
| `width` | `1000` |

Seeds: `[1, 2, 3, 4, 5]`

## Arms

| arm | seeds complete | status |
|---|---|---|
| `lr_0.003` | 0/5 | 0/5 done |
| `lr_0.01` | 0/5 | 0/5 done |
| `lr_0.015` | 0/5 | 0/5 done |
| `lr_0.025` | 0/5 | 0/5 done |
| `lr_0.05` | 0/5 | 0/5 done |
| `lr_0.1` | 0/5 | 0/5 done |
| `lr_0.3` | 0/5 | 0/5 done |

## Phases

0. `sweep` — 7 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S2_lr_frontier --tmux
```
