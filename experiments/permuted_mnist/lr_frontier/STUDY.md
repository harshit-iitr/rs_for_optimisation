# S2_lr_frontier

**Question.** Does lowering the step size buy retention, and is the relationship monotone?

**Supports.** Paper claim 1. Becomes the reference frontier for claim 2.

**Status.** INCOMPLETE (failures present) — 13 complete, 8 failed, 0 not started, of 21 planned runs.

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
| `spectral_every` | `5` |
| `track_drift` | `True` |
| `width` | `1000` |

Seeds: `[1, 2, 3]`

## Arms

| arm | seeds complete | status |
|---|---|---|
| `lr_0.003` | 2/3 | FAILED: seed_1 |
| `lr_0.01` | 2/3 | FAILED: seed_1 |
| `lr_0.015` | 0/3 | FAILED: seed_1, seed_2, seed_3 |
| `lr_0.025` | 2/3 | FAILED: seed_1 |
| `lr_0.05` | 2/3 | FAILED: seed_2 |
| `lr_0.1` | 2/3 | FAILED: seed_3 |
| `lr_0.3` | 3/3 | complete |

## Phases

0. `sweep` — 7 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S2_lr_frontier --tmux
```
