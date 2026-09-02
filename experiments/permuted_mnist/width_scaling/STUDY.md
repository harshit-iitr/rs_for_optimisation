# S5_width_scaling

**Question.** Does the penalty's advantage grow with width?

**Supports.** Supporting. Round 1-4's version lacked per-width baselines at the canonical budget.

**Status.** planned — 0 complete, 0 failed, 24 not started, of 24 planned runs.

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
| `width` | `256` |

Seeds: `[1, 2, 3]`

## Arms

| arm | seeds complete | status |
|---|---|---|
| `width_1000\baseline` | 0/3 | 0/3 done |
| `width_1000\penalty` | 0/3 | 0/3 done |
| `width_2048\baseline` | 0/3 | 0/3 done |
| `width_2048\penalty` | 0/3 | 0/3 done |
| `width_256\baseline` | 0/3 | 0/3 done |
| `width_256\penalty` | 0/3 | 0/3 done |
| `width_512\baseline` | 0/3 | 0/3 done |
| `width_512\penalty` | 0/3 | 0/3 done |

## Phases

0. `sweep` — 8 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S5_width_scaling --tmux
```
