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

| λ | n | test_acc | prev_only | vs λ=0 (paired) | radial_excess | weight_norm |
|---|---|---|---|---|---|---|
| 0 | 3 | 0.9585 ± 0.0011 | 0.2067 ± 0.0024 | — | +18.13 | 50.57 |
| 1e-4 | 3 | 0.9594 ± 0.0006 | 0.2340 ± 0.0038 | +0.027, t=11.1, p=8.0e-03, 3/3 | +17.50 | 50.10 |
| **3e-3** | **3** | 0.9597 ± 0.0002 | **0.3416 ± 0.0020** | **+0.135, t=55.2, p=3.3e-04, 3/3** | +11.48 | 45.24 |
| 6e-3 | 3 | 0.9607 ± 0.0003 | 0.3408 ± 0.0052 | +0.134, t=40.4, p=6.1e-04, 3/3 | +9.19 | 43.34 |
| 1e-2 | 3 | 0.9610 ± 0.0012 | 0.3367 ± 0.0054 | +0.130, t=29.7, p=1.1e-03, 3/3 | +7.23 | 41.67 |
| 0.1 | 2 | 0.9630 ± 0.0001 | 0.3229 ± 0.0027 | +0.116, n=2 | −0.65 | 33.73 |
| 1.0 | 2 | 0.9619 ± 0.0007 | 0.2968 ± 0.0033 | +0.090, n=2 | −0.15 | 31.46 |
| 10.0 | 3 | 0.9624 ± 0.0013 | 0.2910 ± 0.0018 | +0.084, t=56.4, p=3.1e-04, 3/3 | −0.02 | 29.84 |

The optimum is a **plateau over λ ∈ [0.003, 0.006]**, not a sharp peak: 0.3416 vs
0.3408 is well inside seed noise.

**Leg 4a — interior optimum: yes.** Retention peaks at λ≈0.003 and declines
either side. `radial_excess` falls monotonically from +18.1 through zero, so the
constraint visibly engages across the whole sweep and the optimum sits well
before it is enforced.

**Leg 4b — the constraint limit gives nothing, on BOTH limit arms.** Both reach
`radial_excess` = 0.0000, i.e. the constraint is exactly enforced:

| limit arm | n | test_acc | prev_only | vs λ=0 |
|---|---|---|---|---|
| `limit_ste` (straight-through, the Round 1–4 variant) | 3 | 0.9588 ± 0.0003 | 0.2049 ± 0.0025 | −0.002, p=0.30, 1/3 |
| **`limit_tangential`** (true projection, exact Jacobian) | 3 | 0.9580 ± 0.0012 | **0.1994 ± 0.0024** | **−0.007, p=0.078, 0/3** |

The true limit is if anything **slightly worse than no penalty at all**, losing on
all three seeds. So the entire +0.135 benefit lives at finite λ and vanishes
completely once the constraint is enforced. This is the counterintuitive result
the paper is built around, and it now rests on the correct arm rather than on the
straight-through estimator.

### S2 learning-rate frontier — n=2 per point, lr=0.3 failed entirely

| lr | n | test_acc | prev_only | weight_norm |
|---|---|---|---|---|
| 0.003 | 2 | 0.8675 | 0.2624 | 44.86 |
| 0.010 | 2 | 0.9192 | 0.2245 | 45.09 |
| 0.025 | 2 | 0.9454 | 0.2306 | 45.74 |
| 0.050 | 2 | 0.9565 | 0.2375 | 47.24 |
| 0.100 | 2 | 0.9579 | 0.2068 | 50.58 |
| **0.300** | **3** | **0.9649** | **0.1536** | 68.34 |

Spearman ρ=−0.77, p=0.072. Retention falls as lr rises and accuracy rises as lr
rises: the step-size knob trades one axis for the other, moving *along* a
frontier. **This revises the working thesis** — the archived Round 1–4 hint that
lowering lr made retention *worse* does not reproduce at 150 tasks. The paper's
leg 1 should be stated as "the step-size knob only trades along the frontier",
which is what leg 2 then contrasts against.

### Leg 2 — does the penalty curve lie above the lr frontier?

| λ | test_acc | prev_only | best lr-matched | its test | its prev | gap |
|---|---|---|---|---|---|---|
| 3e-3 | 0.9597 | 0.3416 | 0.1 | 0.9579 | 0.2068 | **+0.1348** |
| 6e-3 | 0.9607 | 0.3408 | 0.1 | 0.9579 | 0.2068 | +0.1340 |
| 1e-2 | 0.9610 | 0.3367 | 0.1 | 0.9579 | 0.2068 | +0.1299 |
| 0.1 | 0.9630 | 0.3229 | 0.3 | 0.9649 | 0.1536 | **+0.1693** |
| 1.0 | 0.9619 | 0.2968 | 0.3 | 0.9649 | 0.1536 | +0.1432 |
| 10.0 | 0.9624 | 0.2910 | 0.3 | 0.9649 | 0.1536 | +0.1374 |

**Yes — above at 7 of 8 λ values, max vertical gap +0.169.** With lr=0.3 in place
the frontier now reaches *higher* accuracy (0.9649) than any λ arm, so the
comparison is conservative: the penalty arms are matched against an lr point that
beats them on the plasticity axis and still loses 0.17 on retention.

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

- λ=3e-4, 1e-3, 3e-2 have **0 complete seeds**, leaving gaps in the curve between
  1e-4 and 3e-3 and between 1e-2 and 0.1.
- λ=0.1 and λ=1.0 are **n=2**; every other λ point is n=3.
- S2 is **n=2** at every lr except 0.3.
- `iso_wnorm` is n=3 in both regimes; its clipped-regime acceptance test cannot
  run because two target runs predate the weight-norm trace.
