# Config Audit

*(Note: Config hashes and git shas were not logged by the training scripts in previous rounds. The following configurations have been meticulously audited and reconstructed from `src/sweep.py` and the shell scripts.)*

| Table | Epochs/Task | LR | N Tasks | Width | Depth | Optimizer | Activation | N Seeds | Batch Size | Config Hash |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 10 | 0.1 | 50 | 1000 | 3 | sgd | relu | 5 | 128 | N/A |
| R1 | 1 | 0.1 | 150 | 1000 | 3 | sgd | relu | 3 | 128 | N/A |
| R2 | 1 | 0.1 | 150 | 1000 | 3 | sgd | relu | 3 | 128 | N/A |
| R3 | 10 | 0.1 | 150 | 1000 | 3 | sgd | relu | 5 | 128 | N/A |
| R4 | 1 | 0.1 | 150 | 256-2048 | 3 | sgd | relu | 3 | 128 | N/A |
| R5 | 1 | 0.1 | 150 | 1000 | 3 | sgd | relu | 3 | 128 | N/A |
| R6 | 1 | 0.01 | 150 | 1000 | 3 | adam/sgd | relu | 3 | 128 | N/A |
| S5.1 | 1 | 0.01 | 100 | 256 | 2 | sgd | relu | 5 | 128 | N/A |
| S5.3 | 1 | 0.01 | 10 | convnet | convnet | sgd | relu | 3 | 128 | N/A |
| A1 | 10 | 0.01/0.1 | 300 | 1000 | 3 | sgd | relu | 3 | 128 | N/A |
| B4 | 10 | 0.1 | 50 | 1000 | 3 | sgd | relu | 5 | 128 | N/A |

## Canonical Config Recommendation

The audit reveals significant fragmentation:
- **R1 and R2** (the core stiffness curve and baselines) used `epochs=1` and `n_tasks=150`.
- **C1, R3, B4, A1** used `epochs=10`, typically on 50 or 150 tasks.

**Recommendation:** The canonical config MUST be the one used for **R1 and R2** (`epochs=1, lr=0.1, n_tasks=150, width=1000, depth=3`). 
Reasoning:
1. R1 and R2 contain the most computationally expensive and comprehensive baseline sweeps (EWC, SI, MAS, L2, etc.). Rerunning them at `epochs=10` would take days of compute.
2. The catastrophic forgetting framing holds perfectly well at a fixed, constrained optimization budget (`epochs=1`). We have already established (via A1b) that plasticity loss is minimal even at 1 epoch/task, and that retention is the dominant failure mode. 

Therefore, for T2c, we will rerun C1 (Isotropic Control and Matched BP) at the canonical config: `epochs=1, lr=0.1, n_tasks=150`.
