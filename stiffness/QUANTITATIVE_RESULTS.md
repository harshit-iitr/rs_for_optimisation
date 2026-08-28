# Quantitative Results: Stiffness Campaign (R1-R6)

All numbers are reported directly from the analysis scripts and raw output data without interpretation. 

## R1: Retention Sweep
**Source:** `results/R3_R1_*` (parquet files averaged across 5 seeds for tasks 130-149).
**Aggregation Script:** `analysis/r1_r3_analysis.py`

| Config | Current Task Acc (mean) | Previous Tasks Retention (mean) |
|---|---|---|
| λ=0.0 (BP) | 0.9583 | 0.2097 |
| λ=0.01 | 0.9609 | 0.3394 |
| λ=0.03 | 0.9620 | 0.3342 |
| λ=0.10 | 0.9630 | 0.3275 |
| λ=0.30 | 0.9622 | 0.3129 |
| λ=1.00 | 0.9627 | 0.2984 |
| λ=10.0 | 0.9630 | 0.2957 |
| λ=inf (Hard Proj) | 0.9589 | 0.2116 |


## R3: Drift Metrics 
**Source:** `results/R3_R1_*` (parquet files averaged across 5 seeds for tasks 130-149, measured on fixed task-0 probe).
**Aggregation Script:** `analysis/r1_r3_analysis.py`

| Config | Cosine Sim | Relative Drift (||Δh||/||h||) | Subspace Overlap (mean of top-50 singular values) |
|---|---|---|---|
| λ=0.0 (BP) | 0.9833 | 0.1450 | 0.9478 |
| λ=0.01 | 0.9788 | 0.1847 | 0.9764 |
| λ=0.03 | 0.9816 | 0.1773 | 0.9764 |
| λ=0.10 | 0.9835 | 0.1725 | 0.9753 |
| λ=0.30 | 0.9802 | 0.1893 | 0.9748 |
| λ=1.00 | 0.9784 | 0.1990 | 0.9745 |
| λ=10.0 | 0.9804 | 0.1914 | 0.9735 |
| λ=inf (Hard Proj) | 0.9893 | 0.1249 | 0.9516 |


## R2: Baselines
**Source:** `results/R3_R2_*`
**Output Data:** `results/r2_stats.csv` (averaged across 5 seeds for tasks 130-149).

| Arm Name | Current Task Acc (mean ± std) | Previous Tasks Retention (mean ± std) |
|---|---|---|
| bp | 0.958675 ± 0.001092 | 0.211767 ± 0.003403 |
| cbp | 0.958668 ± 0.000484 | 0.211651 ± 0.004306 |
| l2 | 0.967045 ± 0.000736 | 0.123352 ± 0.000783 |
| l2_init | 0.951325 ± 0.000843 | 0.112338 ± 0.000598 |
| ln_l2 | 0.969505 ± 0.000895 | 0.164705 ± 0.002522 |
| mas | 0.408550 ± 0.014925 | 0.398190 ± 0.004711 |
| rs ($\lambda=0.03$) | 0.961985 ± 0.000861 | 0.331895 ± 0.005269 |
| si | 0.929762 ± 0.001625 | 0.400273 ± 0.002684 |
| sp | 0.946105 ± 0.002089 | 0.123695 ± 0.001269 |
| ewc_10000 | 0.748060 ± 0.002427 | 0.437495 ± 0.003650 |
| rs_ewc_10000 | 0.744466 ± 0.004352 | 0.476748 ± 0.010758 |


## R4: Width Scaling
**Source:** `results/R3_R4_*`
**Output Data:** `results/r4_stats.csv`

| Width | BP Retention | RS ($\lambda=0.03$) Retention | Delta (RS - BP) |
|---|---|---|---|
| 256 | 0.153706 | 0.181019 | 0.027313 |
| 512 | 0.175392 | 0.237238 | 0.061846 |
| 1000 | 0.212070 | 0.336204 | 0.124134 |
| 2048 | 0.294850 | 0.425487 | 0.130636 |


## R5 (S3): Diagnostic Race Correlation
**Source:** `results/R3_R1_*` & `results/R3_R2_*`
**Aggregation Script:** `analysis/r5_headroom_race.py` 
*(Spearman rank correlation `r` vs `normalized_gain`)*

| Layer | Metric | `r` | `p_value` |
|---|---|---|---|
| **Layer 0** | readiness | -0.248079 | 0.000417 |
| Layer 0 | radial_excess | -0.174568 | 0.015319 |
| Layer 0 | phi_rad_tilde | -0.172886 | 0.014997 |
| Layer 0 | eff_rank | -0.023661 | 0.741683 |
| **Layer 1** | stable_rank | 0.290402 | 0.000031 |
| Layer 1 | eff_rank | 0.281762 | 0.000055 |
| Layer 1 | readiness | -0.248079 | 0.000417 |
| Layer 1 | phi_rad_tilde | 0.110792 | 0.121259 (Not significant) |
| **Layer 2** | eff_rank | 0.320167 | 0.000004 |
| Layer 2 | stable_rank | 0.278464 | 0.000068 |
| Layer 2 | readiness | -0.248079 | 0.000417 |
| Layer 2 | phi_rad_tilde | 0.026370 | 0.713336 (Not significant) |


## R6: Optimizers
**Source:** `results/R3_R6_*`
**Output Data:** `results/r6_stats.csv`

| Optimizer | λ | Current Task Acc (mean ± std) | Previous Tasks Retention (mean ± std) |
|---|---|---|---|
| sgd | 0.0 | 0.918930 ± 0.000797 | 0.230033 ± 0.002820 |
| sgd | 0.03 | 0.927995 ± 0.000502 | 0.329880 ± 0.004436 |
| sgd_momentum | 0.0 | 0.958795 ± 0.000574 | 0.208206 ± 0.001887 |
| sgd_momentum | 0.03 | 0.961840 ± 0.000466 | 0.334445 ± 0.002110 |
| adam | 0.0 | 0.116140 ± 0.000594 | 0.116569 ± 0.000649 |
| adam | 0.03 | 0.546345 ± 0.057203 | 0.104056 ± 0.000734 |
| adamw | 0.0 | 0.116106 ± 0.000681 | 0.116385 ± 0.000579 |
| adamw | 0.03 | 0.541675 ± 0.050869 | 0.104104 ± 0.000287 |
*(Note: Adam/AdamW were run at lr=0.1 which causes network failure under Adam's preconditioning, hence the ~11-54% accuracy limit).*

## S5.1: Second Benchmark (Rotating MNIST)
**Source:** `results/S5_1_rot_*`

| Config | Current Task Acc (mean ± std) | Previous Tasks Retention (mean ± std) |
|---|---|---|
| λ=0.0 (BP) | 0.9164 ± 0.0012 | 0.4765 ± 0.0047 |
| λ=1.0 | 0.9193 ± 0.0013 | 0.4245 ± 0.0070 |
| λ=10.0 | 0.9158 ± 0.0004 | 0.4055 ± 0.0071 |
*(Note: Rotating MNIST is a continuous distribution shift, not discrete tasks, meaning classical retention metrics behave differently, but accuracy is maintained).*

## S5.3: ReLU vs LeakyReLU
**Source:** `results/S5_3_*`

| Activation | λ | Current Task Acc (mean ± std) | Previous Tasks Retention (mean ± std) |
|---|---|---|---|
| leaky_relu | 0.0 | 0.7689 ± 0.0023 | 0.3363 ± 0.0068 |
| leaky_relu | 1.0 | 0.7789 ± 0.0010 | 0.4163 ± 0.0076 |
| relu | 0.0 | 0.7682 ± 0.0025 | 0.3369 ± 0.0068 |
| relu | 1.0 | 0.7782 ± 0.0008 | 0.4177 ± 0.0076 |
*(Note: Refutes prior internal claim. Standard ReLU does not collapse under RS; it performs identically to LeakyReLU).*
