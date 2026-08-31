# S3_stiffness_curve

**Question.** Where is the retention optimum in penalty strength, and does the benefit vanish once the constraint is enforced?

**Supports.** Paper claims 2 and 4; supplies the data for the equilibrium-law fit (S4).

**Status.** INCOMPLETE (failures present) — 28 complete, 11 failed, 0 not started, of 39 planned runs.

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
| `track_drift` | `True` |
| `width` | `1000` |

Seeds: `[1, 2, 3]`

## Arms

| arm | seeds complete | status |
|---|---|---|
| `lambda_0.0000` | 3/3 | complete |
| `lambda_0.0001` | 3/3 | complete |
| `lambda_0.0003` | 0/3 | FAILED: seed_1, seed_2, seed_3 |
| `lambda_0.0010` | 0/3 | FAILED: seed_1, seed_2, seed_3 |
| `lambda_0.0030` | 3/3 | complete |
| `lambda_0.0060` | 3/3 | complete |
| `lambda_0.0100` | 3/3 | complete |
| `lambda_0.0300` | 0/3 | FAILED: seed_1, seed_2, seed_3 |
| `lambda_0.1000` | 2/3 | FAILED: seed_3 |
| `lambda_1.0000` | 2/3 | FAILED: seed_3 |
| `lambda_10.0000` | 3/3 | complete |
| `limit_ste` | 3/3 | complete |
| `limit_tangential` | 3/3 | complete |

## Phases

0. `sweep` — 13 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S3_stiffness_curve --tmux
```
