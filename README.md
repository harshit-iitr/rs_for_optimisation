# Activation-geometry constraints and the plasticity–stability frontier

Round 5. A penalty that softly pulls hidden activations toward a fixed-radius
hypersphere, studied as an **optimization** question: does constraining the
geometry of activations move the plasticity–stability frontier in a way that
adjusting the optimizer's step size cannot?

## Layout

```
src/            trainer, penalty, metrics, study registry, launcher
tests/          unit and regression tests (run: PYTHONPATH=. pytest tests/)
experiments/    one directory per study; see experiments/README.md
  _archive/     Rounds 1-4, frozen. Read its README before touching anything there
data/           MNIST, CIFAR (gitignored, re-downloadable)
latex/          write-up
```

## Running

```bash
PYTHONPATH=. python3 -m src.launch --study S1_isotropic_control --measure   # time + memory first
PYTHONPATH=. python3 -m src.launch --study S1_isotropic_control --dry-run
PYTHONPATH=. python3 -m src.launch --study S1_isotropic_control --tmux      # detached, resumable
PYTHONPATH=. python3 -m src.docs                                            # refresh STUDY.md + index
```

The launcher is resumable (a completed run is skipped via its own `config.json`),
gates concurrency on measured free GPU memory, and records failures in
`_launch_report.json` rather than silently skipping them.

## What changed from Rounds 1–4, and why

The audit (`experiments/_archive/round1_4_stiffness/results/REPO_AUDIT.md`) found
that the load-bearing measurements lived inline in the training loop, untested,
with meanings that shifted as the file was edited — eight logging schemas, at
least five trainer versions, and no run recording its own configuration.

- Every measurement now lives behind a tested function in `src/metrics/`.
- **Gradient clipping and gradient scaling can no longer fight each other.** The
  ordering is explicit, asserted at runtime, and covered by a regression test
  (`tests/test_isotropic.py::test_clip_after_scale_destroys_the_match_THE_ROUND_1_4_BUG`).
- The isotropic control matches **realized per-step gradient magnitude**, per
  seed, across **all** parameters, and must pass a pre-registered acceptance test
  before any retention number is looked at.
- `prev_only_acc` (previous tasks only) is the retention metric; the Round 1–4
  quantity is kept under its accurate name `avg_seen_acc`.
- Absolute drift is logged alongside relative drift, which divides by a quantity
  the penalty shrinks.
- The hard-constraint arm is a **true tangential projection**; the Round 1–4
  straight-through estimator is retained as a separately-named arm.
- Every run writes `config.json` before training starts.
- Nothing is ever averaged across layers.
