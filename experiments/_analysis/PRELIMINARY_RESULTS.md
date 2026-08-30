# Round 5 — Preliminary Results

**Status: PRELIMINARY. Nothing here is publishable.** Seed counts vary per arm
(some n=1), some arms mix `spectral_every` settings, and no study is complete.
Every number below is stamped with its true n. Regenerate with:

```bash
PYTHONPATH=. python3 analysis/s1_isotropic.py   --preliminary
PYTHONPATH=. python3 analysis/s2_s3_frontier.py --preliminary
PYTHONPATH=. python3 analysis/s4_equilibrium.py --preliminary
```

Canonical config throughout: permuted MNIST, 150 tasks, 10 epochs/task, MLP
width 1000 depth 3, ReLU, SGD lr 0.1, batch 256, probe 2000.
Retention metric is `prev_only_acc` (previous tasks only, current task excluded),
final-20-task window, layer 0.

---

## Leg 3 — mechanism: direction, not magnitude (S1)

Two clipping regimes. The distinction matters more than expected.

### clipped (`clip_norm=0.5`, canonical) — n=5 except iso_wnorm n=3

Clip binds on **89%** of steps and pins *both* arms to a global gradient norm of
exactly 0.500, so the **global magnitude confound is 0.00%**. Only the per-layer
confound is real here (up to 10.7% at layer 2).

| arm | test_acc | prev_only_acc | weight_norm | vs baseline |
|---|---|---|---|---|
| baseline | 0.9587 ± 0.0011 | 0.2064 ± 0.0034 | 50.57 | — |
| penalty λ=0.003 | 0.9595 ± 0.0005 | **0.3419 ± 0.0017** | 45.26 | +0.1365, t=+50.7, p=3.9e-04, 3/3 |
| isotropic per-layer | 0.9600 ± 0.0004 | 0.2136 ± 0.0076 | 50.59 | +0.0072, p=0.091, 4/5 |
| isotropic global | 0.9599 ± 0.0007 | 0.2087 ± 0.0068 | 50.18 | +0.0022, p=0.34, 3/5 |

Magnitude matching recovers **5.3%** (per-layer) / **1.7%** (global) of the gain.
All controls pass their acceptance test exactly (median |log ratio| = 0.00e+00).

### loose (`clip_norm=10`) — n=3, the informative regime

Clip binds on only 46% of steps, so the magnitude confound is **real**: 20.1%
globally, up to 40.7% at the head. This is where a magnitude-matched control has
something to remove.

| arm | test_acc | prev_only_acc | weight_norm | recovers |
|---|---|---|---|---|
| baseline | 0.9644 ± 0.0007 | 0.2073 ± 0.0045 | 59.05 | — |
| penalty λ=0.003 | 0.9632 ± 0.0007 | **0.3047 ± 0.0055** | 48.01 | — (+0.1006, t=+47.8, p=4.4e-04, 3/3) |
| isotropic per-layer | 0.9643 ± 0.0007 | 0.2063 ± 0.0047 | 59.05 | **−1.0%** |
| isotropic global | 0.9643 ± 0.0005 | 0.2000 ± 0.0056 | 57.60 | **−7.4%** |
| iso weight-norm | 0.9671 ± 0.0007 | 0.2231 ± 0.0134 | **48.25** | **+16.3%** |

**The key line is the last one.** The weight-space control reproduces the
penalty's weight norm almost exactly (48.25 vs 48.01) and recovers 16.3% of the
retention gain — more than either gradient control, but leaving **~84%
unexplained**. So "the penalty just parks the network at a smaller norm" is a
real but minor part of the story, and it is now measured rather than argued.

---

## Legs 1 and 2 — the two curves (S2, S3)

### S3 stiffness curve — n=3 except λ=0.003 and λ=0.006 (n=1)

| λ | n | test_acc | prev_only | radial_excess | weight_norm |
|---|---|---|---|---|---|
| 0 | 3 | 0.9585 | 0.2067 | +18.13 | 50.57 |
| 1e-4 | 3 | 0.9594 | 0.2340 | +17.50 | 50.10 |
| **3e-3** | **1** | 0.9599 | **0.3439** | +11.43 | 45.20 |
| 6e-3 | 1 | 0.9605 | 0.3375 | +9.09 | 43.32 |
| 1e-2 | 3 | 0.9610 | 0.3367 | +7.23 | 41.67 |
| 0.1 | 2 | 0.9630 | 0.3229 | −0.65 | 33.73 |
| 1.0 | 2 | 0.9619 | 0.2968 | −0.15 | 31.46 |
| 10.0 | 3 | 0.9624 | 0.2910 | −0.02 | 29.84 |

**Leg 4a — interior optimum: yes.** Retention peaks at λ≈0.003 and declines
either side. `radial_excess` falls monotonically from +18.1 through zero, so the
constraint visibly engages across the whole sweep and the optimum sits well
before it is enforced.

**Leg 4b — the constraint limit gives nothing.** The straight-through limit arm
(`limit_ste`, n=3) reaches `radial_excess` = 0.0000 and retention
**0.2049 ± 0.0025** vs baseline 0.2067 — no benefit (p=0.30, 1/3 wins).
*The true tangential-projection limit arm has not completed and is the one that
matters; `ste` is the Round 1–4 straight-through variant, kept only for
comparison.*

### S2 learning-rate frontier — n=2 per point, lr=0.3 failed entirely

| lr | test_acc | prev_only |
|---|---|---|
| 0.003 | 0.8675 | 0.2624 |
| 0.010 | 0.9192 | 0.2245 |
| 0.025 | 0.9454 | 0.2306 |
| 0.050 | 0.9565 | 0.2375 |
| 0.100 | 0.9579 | 0.2068 |

Not monotone in lr (Spearman ρ=−0.60, p=0.29), and retention is highest at the
*lowest* lr — but at a 9-point accuracy cost. **This revises the working thesis:**
the step-size knob does buy retention, it simply buys it by moving *along* a
frontier, trading accuracy for it. The archived Round 1–4 hint that lowering lr
made retention *worse* does not reproduce at 150 tasks.

### Leg 2 — does the penalty curve lie above the lr frontier?

| λ | test_acc | prev_only | best lr-matched | its prev | gap |
|---|---|---|---|---|---|
| 3e-3 | 0.9599 | 0.3439 | 0.1 | 0.2068 | **+0.1371** |
| 1e-2 | 0.9610 | 0.3367 | 0.1 | 0.2068 | +0.1299 |
| 10.0 | 0.9624 | 0.2910 | 0.1 | 0.2068 | +0.0842 |

**Yes — above at 7 of 8 λ values, max vertical gap +0.137.** Caveat: every λ arm
matches to lr=0.1 because no other lr reaches that accuracy, and lr=0.3 (which
might) failed entirely. The frontier needs its high-accuracy end before this is
solid.

---

## S4 — equilibrium law (free, from S3)

Prediction: log-log slope of radial excess vs λ should be −1.

| layer | n_λ | slope(u*) | ci95 | slope(g_rad)=a | a−1 | naive ok? | refined ok? |
|---|---|---|---|---|---|---|---|
| 0 | 4 | −0.175 | 0.153 | −0.016 | −1.016 | no | no |
| 1 | 6 | −0.931 | 0.305 | +0.076 | −0.924 | **yes** | **yes** |
| 2 | — | too few positive-excess points | | | | | |

Layer 1 matches both the naive −1 and the refined a−1 prediction. Layer 0 matches
neither. Too few λ points to say more; this needs the full S3 grid.

---

## What is missing

| study | complete | planned |
|---|---|---|
| S3 stiffness curve | 21 | 39 |
| S2 lr frontier | 10 | 21 |
| S1 isotropic control | 40 | 50 |
| S5–S9 supporting | 0 | 105 |

Specific gaps that block specific claims:

- `limit_tangential` (the true constraint limit) — **0/3**. Leg 4b currently
  rests on the straight-through variant only.
- `lr_0.3` — **0/3**. The high-accuracy end of the lr frontier, which leg 2 needs.
- λ=0.003 and λ=0.006 are **n=1**, and λ=0.003 is the reported optimum.
- `iso_wnorm` is n=3 in both regimes; its clipped-regime acceptance test cannot
  run because two target runs predate the weight-norm trace.
