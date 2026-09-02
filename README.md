# Radial suppression as an optimization intervention

A soft penalty pulls hidden pre-activations toward a fixed-radius sphere:

```
L_pen(h) = (1/d) * (||h||_2 - sqrt(d))^2
```

We study what it does to optimization dynamics, using a long task sequence as the
measurement instrument. Paper draft in [`latex/`](latex/) (OPT 2026).

## Layout

```
src/            trainer, launcher, study registry, metrics, methods
tests/          regression tests -- run these before trusting a change
analysis/       analysis scripts; analysis/common.py enforces the reporting rules
experiments/    one directory per study; see experiments/README.md
  _archive/     Rounds 1-4, frozen. Read its README before using anything there.
latex/          the paper
opt2026_style/  pristine OPT 2026 template, for diffing against latex/
```

## Getting started

```bash
pip install -r requirements.txt
PYTHONPATH=. python3 -m pytest tests/ -q          # 30 tests, all should pass
PYTHONPATH=. python3 -m src.launch --study S3_stiffness_curve --dry-run
```

MNIST downloads on first run into `data/`, which is gitignored.

## Adding an experiment

Studies are **data**, not code. Add an entry to `STUDIES` in
[`src/studies.py`](src/studies.py) giving the question it answers, which claim it
supports, its seeds, and its arms. The launcher materialises it into
`experiments/<benchmark>/<study>/<arm>/seed_N/`, so a run's path states what it
is and no run-id decoding is needed.

```bash
PYTHONPATH=. python3 -m src.launch --study MY_STUDY --dry-run
PYTHONPATH=. python3 -m src.launch --study MY_STUDY --gpus 0 --concurrency 8 --threads 16 --tmux
PYTHONPATH=. python3 -m src.docs                  # regenerate README + STUDY.md from run status
```

The launcher is resumable, gates concurrency on measured free GPU memory, retries
runs that die, and records failures in `_launch_report.json`. A run completed
under a configuration that no longer matches the study is treated as stale and
re-run, so a study cannot silently mix configurations.

## Conventions that are not optional

These exist because Rounds 1-4 violated each of them and produced results that
had to be withdrawn. The audit is at
`experiments/_archive/round1_4_stiffness/results/REPO_AUDIT.md`.

1. **Every run writes `config.json` before training starts**, including every
   constant that would otherwise be implicit, a schema version and the code
   revision. Nothing is reconstructed from a directory name.
2. **An arm with missing seeds is not reported.** `analysis/common.py` raises
   rather than averaging the survivors. Pass `--preliminary` to override, and
   every row is then stamped with its true `n`.
3. **Per-layer metrics stay per layer.** Never averaged across layers.
4. **Comparisons report test statistic, p, n and sign split**, paired where seeds
   align.
5. **Identical values across arms to four decimals are a bug signal.** Two
   optimizer arms once matched bitwise because both had weight decay zero.
6. **An arm that fails its own acceptance test is not reported at all.**
7. Task-loss gradient only in the radial-energy metric; there is a regression
   test for this.

## Regenerating figures and analyses

```bash
PYTHONPATH=. python3 analysis/figures.py                       # paper figures from run data
PYTHONPATH=. python3 analysis/s1_isotropic.py --preliminary
PYTHONPATH=. python3 analysis/s2_s3_frontier.py --preliminary
PYTHONPATH=. python3 analysis/characterization.py              # full metric battery
```

## Not in this repository

- `data/` -- MNIST and CIFAR, re-downloadable.
- `*.pt` checkpoints.
- `grad_trace.npz` -- per-step gradient and weight-norm traces, 335 MB. Needed
  only to build the isotropic control arms; regenerate with `--log_grad_trace`.
  The launcher runs the target arms before any arm that consumes them.
