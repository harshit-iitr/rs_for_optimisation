# S7_optimizers

**Question.** Does the effect survive momentum, and does it survive adaptive preconditioning?

**Supports.** Supporting. Single code generation; lr stated per optimizer.

**Status.** planned — 0 complete, 0 failed, 18 not started, of 18 planned runs.

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
| `adam\baseline` | 0/3 | 0/3 done |
| `adam\penalty` | 0/3 | 0/3 done |
| `sgd\baseline` | 0/3 | 0/3 done |
| `sgd\penalty` | 0/3 | 0/3 done |
| `sgd_momentum\baseline` | 0/3 | 0/3 done |
| `sgd_momentum\penalty` | 0/3 | 0/3 done |

## Phases

0. `sweep` — 6 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S7_optimizers --tmux
```
