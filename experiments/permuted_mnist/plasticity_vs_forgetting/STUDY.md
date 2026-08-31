# S9_plasticity_vs_forgetting

**Question.** Which failure mode does this benchmark exhibit at the canonical config -- loss of plasticity or forgetting?

**Supports.** Setup. Establishes what the paper is measuring.

**Status.** INCOMPLETE (failures present) — 0 complete, 6 failed, 0 not started, of 6 planned runs.

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
| `n_tasks` | `300` |
| `optimizer` | `sgd` |
| `probe_size` | `2000` |
| `spectral_every` | `5` |
| `width` | `1000` |

Seeds: `[1, 2, 3]`

## Arms

| arm | seeds complete | status |
|---|---|---|
| `baseline_300tasks` | 0/3 | FAILED: seed_1, seed_2, seed_3 |
| `penalty_300tasks` | 0/3 | FAILED: seed_1, seed_2, seed_3 |

## Phases

0. `sweep` — 2 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S9_plasticity_vs_forgetting --tmux
```
