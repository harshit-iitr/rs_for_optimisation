# S3_stiffness_curve

**Question.** Where is the retention optimum in penalty strength, and does the benefit vanish once the constraint is enforced?

**Supports.** Paper claims 2 and 4; supplies the data for the equilibrium-law fit (S4).

**Status.** planned — 0 complete, 0 failed, 75 not started, of 75 planned runs.

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
| `track_drift` | `True` |
| `width` | `1000` |

Seeds: `[1, 2, 3, 4, 5]`

## Arms

| arm | seeds complete | status |
|---|---|---|
| `lambda_0.0000` | 0/5 | 0/5 done |
| `lambda_0.0001` | 0/5 | 0/5 done |
| `lambda_0.0003` | 0/5 | 0/5 done |
| `lambda_0.0010` | 0/5 | 0/5 done |
| `lambda_0.0030` | 0/5 | 0/5 done |
| `lambda_0.0060` | 0/5 | 0/5 done |
| `lambda_0.0100` | 0/5 | 0/5 done |
| `lambda_0.0300` | 0/5 | 0/5 done |
| `lambda_0.1000` | 0/5 | 0/5 done |
| `lambda_0.3000` | 0/5 | 0/5 done |
| `lambda_1.0000` | 0/5 | 0/5 done |
| `lambda_10.0000` | 0/5 | 0/5 done |
| `lambda_3.0000` | 0/5 | 0/5 done |
| `limit_ste` | 0/5 | 0/5 done |
| `limit_tangential` | 0/5 | 0/5 done |

## Phases

0. `sweep` — 15 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S3_stiffness_curve --tmux
```
