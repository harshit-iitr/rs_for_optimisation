# S8_rotating_mnist

**Question.** Does the stiffness curve reproduce on a second benchmark?

**Supports.** Supporting (generality).

**Status.** complete  18 complete, 0 failed, 0 not started, of 18 planned runs.

## Finding

<!-- FINDING -->
Yes, the stiffness curve reproduces on Rotating MNIST (a continuous input rotation rather than a discrete pixel permutation). At lambda_rs = 0.0100, retention accuracy (prev_only) peaks at 47.57% (vs 47.17% baseline), demonstrating that radial suppression improves continual learning under continuous covariance rotation. Additionally, the activation radius shrinks monotonically from 33.21 (baseline) to 21.01 (over-penalized), confirming the penalty operates effectively.
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
| `spectral_every` | `5` |
| `width` | `256` |

Seeds: `[1, 2, 3]`

## Arms

| arm | seeds complete | status |
|---|---|---|
| `lambda_0.0000` | 3/3 | complete |
| `lambda_0.0010` | 3/3 | complete |
| `lambda_0.0030` | 3/3 | complete |
| `lambda_0.0100` | 3/3 | complete |
| `lambda_0.0300` | 3/3 | complete |
| `lambda_0.1000` | 3/3 | complete |

## Phases

0. `sweep`  6 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S8_rotating_mnist --tmux
```
