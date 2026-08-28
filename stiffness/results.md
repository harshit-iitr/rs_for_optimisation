# Final Raw Results & Metrics

## 1. S3 Diagnostic Race (Baseline vs Penalized)

Correlating metric at task $t$ with first epoch gain at task $t+1$.

### Correlations: lambda=0 only (n=745)

| Metric | Spearman r | Spearman p-value | Kendall tau | Kendall p-value |
| :--- | :--- | :--- | :--- | :--- |
| **phi_rad_tilde** | 0.014 | 7.0e-01 | 0.009 | 7.3e-01 |
| **eff_rank** | 0.035 | 3.4e-01 | 0.023 | 3.5e-01 |
| **stable_rank** | 0.066 | 7.1e-02 | 0.044 | 7.2e-02 |
| **dead_frac** | -0.012 | 7.3e-01 | -0.008 | 7.4e-01 |
| **dormant_frac** | -0.016 | 6.7e-01 | -0.009 | 7.0e-01 |
| **weight_norm** | -0.015 | 6.9e-01 | -0.009 | 7.0e-01 |
| **readiness** | -0.050 | 1.7e-01 | -0.033 | 1.8e-01 |

### Correlations: Pooled across all S1/S2 arms (n=9685)

| Metric | Spearman r | Spearman p-value | Kendall tau | Kendall p-value |
| :--- | :--- | :--- | :--- | :--- |
| **phi_rad_tilde** | 0.274 | 2.0e-166 | 0.170 | 8.0e-139 |
| **eff_rank** | -0.214 | 4.7e-101 | -0.116 | 3.5e-65 |
| **stable_rank** | -0.161 | 4.6e-57 | -0.084 | 4.4e-35 |
| **dead_frac** | 0.184 | 6.6e-75 | 0.130 | 1.5e-72 |
| **dormant_frac** | 0.310 | 1.3e-214 | 0.212 | 1.2e-212 |
| **weight_norm** | 0.405 | 0.0e+00 | 0.249 | 1.9e-294 |
| **readiness** | -0.027 | 6.9e-03 | -0.024 | 4.6e-04 |

---

## 2. Full Suite Sweep Metrics (Final Averages)

*Note: S5.2 (Split CIFAR-100) crashed early due to a batch size assertion on the tiny test set and is omitted.*

### S4.3: Width Scaling (RS lambda=1.0)

|     width |   avg_test_acc_mean |   avg_test_acc_std |   avg_prev_acc_mean |   avg_prev_acc_std |
|----------:|--------------------:|-------------------:|--------------------:|-------------------:|
|  256.0000 |              0.7196 |             0.0007 |              0.4975 |             0.0034 |
|  512.0000 |              0.7440 |             0.0007 |              0.5122 |             0.0026 |
| 1000.0000 |              0.7756 |             0.0001 |              0.5336 |             0.0037 |
| 2048.0000 |              0.8093 |             0.0011 |              0.5692 |             0.0029 |

### S5.1: Rotating MNIST (MLP depth=2, width=256)

|   lambda_rs |   avg_test_acc_mean |   avg_test_acc_std |   avg_prev_acc_mean |   avg_prev_acc_std |
|------------:|--------------------:|-------------------:|--------------------:|-------------------:|
|      0.0000 |              0.9062 |             0.0004 |              0.6651 |             0.0030 |
|      1.0000 |              0.9068 |             0.0008 |              0.6326 |             0.0023 |
|     10.0000 |              0.8993 |             0.0006 |              0.6083 |             0.0020 |

### S5.3: Activation Functions (Permuted MNIST)

| act_fn     |   lambda_rs |   avg_test_acc_mean |   avg_test_acc_std |   avg_prev_acc_mean |   avg_prev_acc_std |
|:-----------|------------:|--------------------:|-------------------:|--------------------:|-------------------:|
| leaky_relu |      0.0000 |              0.7709 |             0.0012 |              0.4717 |             0.0020 |
| leaky_relu |      1.0000 |              0.7762 |             0.0001 |              0.5327 |             0.0037 |
| relu       |      0.0000 |              0.7700 |             0.0012 |              0.4723 |             0.0020 |
| relu       |      1.0000 |              0.7756 |             0.0001 |              0.5336 |             0.0037 |

### S6.1: Learning Rate Robustness

|     lr | method   |   avg_test_acc_mean |   avg_test_acc_std |   avg_prev_acc_mean |   avg_prev_acc_std |
|-------:|:---------|--------------------:|-------------------:|--------------------:|-------------------:|
| 0.0010 | bp       |              0.2945 |             0.0006 |              0.2689 |             0.0035 |
| 0.0010 | l2_init  |              0.2944 |             0.0006 |              0.2672 |             0.0035 |
| 0.0010 | rs       |              0.2898 |             0.0007 |              0.2669 |             0.0036 |
| 0.0100 | bp       |              0.7705 |             0.0012 |              0.4715 |             0.0023 |
| 0.0100 | l2_init  |              0.7747 |             0.0010 |              0.4489 |             0.0025 |
| 0.0100 | rs       |              0.7755 |           nan      |              0.5347 |           nan      |
| 0.1000 | bp       |              0.9131 |             0.0004 |              0.3920 |             0.0028 |
| 0.1000 | l2_init  |              0.9066 |             0.0008 |              0.2652 |             0.0014 |
| 0.1000 | rs       |              0.9219 |             0.0005 |              0.4694 |             0.0031 |

### S6.2: Optimizer Interaction (AdamW vs SGD)

| optimizer   |   avg_test_acc_mean |   avg_test_acc_std |   avg_prev_acc_mean |   avg_prev_acc_std |
|:------------|--------------------:|-------------------:|--------------------:|-------------------:|
| adamw       |              0.8332 |             0.0040 |              0.1350 |             0.0001 |
| sgd         |              0.7756 |             0.0001 |              0.5336 |             0.0037 |
