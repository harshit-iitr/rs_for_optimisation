# Experiments

Round 5. One study per directory; a run's path states what it is, so no run-id decoding is needed. Every run carries its own `config.json`, written before training starts.

Regenerate this file and every `STUDY.md` with `python3 -m src.docs`. Status is read from the runs themselves, never typed by hand.

| study | question | supports | status | runs |
|---|---|---|---|---|
| [`permuted_mnist/isotropic_control`](permuted_mnist/isotropic_control/STUDY.md) | Is the penalty's retention benefit reproduced by an equal-magnitude, direction-unconstrained intervention? | Paper claim 3 (mechanism: direction vs magnitude). Decisive. | planned | 0/40 |
| [`permuted_mnist/lr_frontier`](permuted_mnist/lr_frontier/STUDY.md) | Does lowering the step size buy retention, and is the relationship monotone? | Paper claim 1. Becomes the reference frontier for claim 2. | planned | 0/35 |
| [`permuted_mnist/stiffness_curve`](permuted_mnist/stiffness_curve/STUDY.md) | Where is the retention optimum in penalty strength, and does the benefit vanish once the constraint is enforced? | Paper claims 2 and 4; supplies the data for the equilibrium-law fit (S4). | planned | 0/75 |
| [`permuted_mnist/width_scaling`](permuted_mnist/width_scaling/STUDY.md) | Does the penalty's advantage grow with width? | Supporting. Round 1-4's version lacked per-width baselines at the canonical budget. | planned | 0/24 |
| [`permuted_mnist/baselines`](permuted_mnist/baselines/STUDY.md) | Where does the penalty sit on the plasticity-stability frontier relative to parameter-regularization baselines, and does it compose? | Supporting (Pareto position and additivity). | planned | 0/65 |
| [`permuted_mnist/optimizers`](permuted_mnist/optimizers/STUDY.md) | Does the effect survive momentum, and does it survive adaptive preconditioning? | Supporting. Single code generation; lr stated per optimizer. | planned | 0/30 |
| [`rotating_mnist/stiffness_curve`](rotating_mnist/stiffness_curve/STUDY.md) | Does the stiffness curve reproduce on a second benchmark? | Supporting (generality). | planned | 0/30 |
| [`permuted_mnist/plasticity_vs_forgetting`](permuted_mnist/plasticity_vs_forgetting/STUDY.md) | Which failure mode does this benchmark exhibit at the canonical config -- loss of plasticity or forgetting? | Setup. Establishes what the paper is measuring. | planned | 0/10 |

## Layout

```
experiments/
  <benchmark>/<study>/STUDY.md          question, config, arms, status, finding
  <benchmark>/<study>/<arm>/seed_N/     config.json, metrics.parquet,
                                        grad_trace.npz, stdout.log
  _analysis/<study>/                    figures, tables, stats
  _archive/                             Round 1-4. Frozen. Never globbed.
```

## Rules

1. Every comparison reports a test statistic, p-value, n and sign split; paired where seeds align.
2. Both axes on every table. `prev_only_acc` is the retention metric; `avg_seen_acc` is the Round 1-4 quantity, kept for comparability.
3. Per-layer metrics stay per layer. Nothing is ever averaged across layers.
4. Every table is headed with its full config, read from `config.json`.
5. Values identical across arms to four decimals are a bug signal and are investigated before anything is reported.
6. No arm is reported with fewer seeds than planned. No `nan` standard deviations.
7. An arm that fails its own acceptance test is not reported at all.
8. Task-loss gradient only in the radial-energy metric.
9. Surprises go to `experiments/ANOMALIES.md`.
