# S1_isotropic_control

**Question.** Is the penalty's retention benefit reproduced by an equal-magnitude or equal-weight-norm intervention that carries no directional information?

**Supports.** Paper claim 3 (mechanism: direction vs magnitude). Decisive.

**Status.** INCOMPLETE (failures present) � 38 complete, 4 failed, 8 not started, of 50 planned runs.

## Finding

<!-- FINDING -->
**Clipped regime: complete (20/20). The penalty's benefit is NOT reproduced by
magnitude matching.** A per-layer magnitude-matched control that passes its
acceptance test *exactly* (median |log ratio| = 0.00e+00 on every seed, every
group) recovers **5.4%** of the penalty's retention gain; the global-norm control
recovers 1.7%. Retention `prev_only_acc`, final 20 tasks, layer 0, n=5, paired:

| arm | test_acc | prev_only_acc | vs baseline |
|---|---|---|---|
| baseline | 0.9587 ± 0.0011 | 0.2064 ± 0.0034 | — |
| penalty λ=0.003 | 0.9595 ± 0.0004 | **0.3413 ± 0.0015** | +0.1349, t=+74.4, p=2.0e-07, 5/5 |
| isotropic per-layer | 0.9600 ± 0.0004 | 0.2136 ± 0.0076 | +0.0072, t=+2.22, p=0.091, 4/5 |
| isotropic global | 0.9599 ± 0.0007 | 0.2087 ± 0.0068 | +0.0022, t=+1.08, p=0.34, 3/5 |

Current-task accuracy is statistically indistinguishable across all four arms
(penalty vs baseline p=0.23), so the retention gain costs no plasticity.

**Two caveats that must travel with this number.**

1. Under clipping the **global** magnitude confound is exactly 0.00% — `clip_grad_norm_(0.5)`
   binds on 89% of steps and pins both arms to a total gradient norm of 0.500.
   A global-norm control therefore matches a quantity that is already equal, and
   its null is uninformative. Only the per-layer arm carries information here,
   because the per-layer confound is real (up to 10.7%, at layer 2). The
   `unclipped` regime exists to test the magnitude hypothesis where the global
   confound is not masked.
2. The control matches realized **gradient magnitude**, which is an intervention.
   It does not reproduce the penalty's weight-norm reduction (baseline 50.57,
   penalty 45.26, control 50.59) — but weight norm is an *outcome* of the
   penalty's directional effect, not a knob, so matching it would beg the
   question. Worth stating explicitly in the paper before a reviewer asks.

Round 1–4 reported −0.6% here from a control that had silently achieved 4% of its
intended match. The qualitative conclusion survives; the evidence for it did not
exist until now.

**Unclipped regime: in progress.**
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
| `spectral_every` | `5` |
| `track_drift` | `True` |
| `width` | `1000` |

Seeds: `[1, 2, 3, 4, 5]`

## Arms

| arm | seeds complete | status |
|---|---|---|
| `clipped\arm_baseline` | 5/5 | complete |
| `clipped\arm_iso_wnorm` | 3/5 | 3/5 done |
| `clipped\arm_isotropic_global` | 5/5 | complete |
| `clipped\arm_isotropic_per_layer` | 5/5 | complete |
| `clipped\arm_penalty_lam0.003` | 3/5 | FAILED: seed_1, seed_3 |
| `loose\arm_baseline` | 5/5 | complete |
| `loose\arm_iso_wnorm` | 3/5 | 3/5 done |
| `loose\arm_isotropic_global` | 3/5 | 3/5 done |
| `loose\arm_isotropic_per_layer` | 3/5 | 3/5 done |
| `loose\arm_penalty_lam0.003` | 3/5 | FAILED: seed_3, seed_5 |

## Phases

0. `clipped_A_targets` � 2 arm(s)
1. `clipped_B_controls` � 3 arm(s)
2. `loose_A_targets` � 2 arm(s)
3. `loose_B_controls` � 3 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S1_isotropic_control --tmux
```
