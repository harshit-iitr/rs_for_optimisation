# S8_rotating_mnist

**Question.** Does the stiffness curve reproduce on a second benchmark?

**Supports.** Supporting (generality).

**Status.** planned — 0 complete, 0 failed, 30 not started, of 30 planned runs.

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
| `dataset` | `rotating_mnist` |
| `depth` | `2` |
| `epochs` | `1` |
| `lr` | `0.01` |
| `n_tasks` | `100` |
| `optimizer` | `sgd` |
| `probe_size` | `2000` |
| `width` | `256` |

Seeds: `[1, 2, 3, 4, 5]`

## Arms

| arm | seeds complete | status |
|---|---|---|
| `lambda_0.0000` | 0/5 | 0/5 done |
| `lambda_0.0010` | 0/5 | 0/5 done |
| `lambda_0.0030` | 0/5 | 0/5 done |
| `lambda_0.0100` | 0/5 | 0/5 done |
| `lambda_0.0300` | 0/5 | 0/5 done |
| `lambda_0.1000` | 0/5 | 0/5 done |

## Phases

0. `sweep` — 6 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S8_rotating_mnist --tmux
```
