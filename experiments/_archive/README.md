# Archive — Rounds 1–4

Frozen. **Nothing in here is trusted, and no analysis script may glob into it.**

Kept because it is the project's record: several of these numbers appear in
`round1_4_stiffness/results/REPO_AUDIT.md`, in the internal reports, and in
`latex/report.tex`. Deleting it would make those unauditable.

## What is here

| path | what |
|---|---|
| `round1_4_stiffness/` | the Round 1–4 codebase, 733 run directories, all reports |
| `round1_4_stiffness/results/REPO_AUDIT.md` | the audit. Read this before using anything here |
| `legacy_experiments/` | an earlier, separate implementation (five per-benchmark trainers) |

## Every archived run now has a `config.json`

Round 1–4 recorded no configuration at all. `src/backfill_archive.py`
reconstructed one for each run from its run-id and the archived launcher
definitions. Every reconstructed config is stamped:

- `"provenance": "reconstructed"` — inferred, **not** recorded at run time
- `"schema_generation"` — fingerprinted from the parquet's column set, the
  archive's only reliable clock (audit §2.1)
- `"has_retention_column"` — false for 108 runs, which therefore cannot speak to
  any retention claim
- `"KNOWN_DEFECT"` — on arms the audit found to be broken
- `"status": "failed"` — on the 47 runs that died before writing output

Five runs are stamped `"provenance": "unrecoverable"`: `S0.2_seed{1,2,3}`,
`S0.3_seed1`, `S5_2_test`. No archived launcher defines them. Do not use them.

## The defects, in one place

Full analysis in the audit; short form so nobody has to rediscover them.

| what | severity |
|---|---|
| `bp_iso` is an invalid control — gradient scaling applied *before* `clip_grad_norm_`, which renormalized it away. Achieved 0.7% of an intended 18.5% match (§3.3.2) | fatal to C2 |
| `cbp` is inert — never effective, and not instantiated by the final trainer at all. Identical to `bp` (§5.1) | fatal to the CBP baseline |
| `adam` and `adamw` runs are **bitwise identical** — `AdamW(wd=0) ≡ Adam(wd=0)`, and `'adam' in 'adamw'` gave both the same lr (§5.3) | fatal to C8 |
| `readiness` is one network-wide value broadcast into every layer row, in 156 of 182 run families (§5.2) | fatal to the R5 table |
| Split-CIFAR-100 never trained — ~19 steps/task under a 0.5 grad clip; `train_acc` is 7% on the task just trained (§5.4) | fatal to all CIFAR results |
| `er` / `l2_er` collapse the network to ~0.11 accuracy (§5.5a); `mas` never trains, so its retention is an artifact (§5.5b) | those arms only |
| The `λ=inf` arm is a straight-through estimator, not the constraint limit (§5.5c) | reframes C4's endpoint |
| `radial_energy` in `legacy_experiments/` is computed on the **total** gradient including the penalty's own purely-radial contribution — penalised arms read 83–101 against a baseline of 0.02 (§3.3.1) | fatal to the Φ_rad diagnostic |

## Comparability

Eight logging schemas and at least five trainer versions are mixed in here. Pairs
that look comparable and are not:

- `B4_*` (lr 0.01) vs `R2_B4_*` (lr 0.1) — 14× difference in `phi_rad_tilde`
- `C1_*` (150 tasks) vs `R2_C1_*` (50 tasks)
- `S1_*`/`S2_*` (1 epoch, lr 0.01) vs `R3_*`/`R4_*` (10 epochs, lr 0.1)
- `R3_R6_opt_sgd*` (schema gen 3) vs `R3_R6_opt_adam*` (schema gen 0) — different
  trainer versions inside one reported table

Each run's `config.json` now makes these visible without decoding a run-id.
