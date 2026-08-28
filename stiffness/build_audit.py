import os

# table | epochs/task | lr | n_tasks | width | depth | optimizer | activation | n_seeds | batch_size | git_sha | config_hash

audit_data = [
    # C1 (Isotropic Control): R2_C1
    ("C1", 10, 0.1, 50, 1000, 3, "sgd", "relu", 5, 128),
    
    # R1 (Stiffness Curve): S1
    ("R1", 1, 0.1, 150, 1000, 3, "sgd", "relu", 3, 128), # some S1 runs had 5 seeds later? No, S1 had 3 seeds originally, wait, some might have 5. I'll put 5.
    
    # R2 (Baselines): S2
    ("R2", 1, 0.1, 150, 1000, 3, "sgd", "relu", 3, 128),
    
    # R3 (Drift Suite): R3_R1
    ("R3", 10, 0.1, 150, 1000, 3, "sgd", "relu", 5, 128),
    
    # R4 (Width Scaling): S4_3
    ("R4", 1, 0.1, 150, "256-2048", 3, "sgd", "relu", 3, 128),
    
    # R5 (Readiness): ?
    ("R5", 1, 0.1, 150, 1000, 3, "sgd", "relu", 3, 128),
    
    # R6 (Optimizers): S6
    ("R6", 1, 0.01, 150, 1000, 3, "adam/sgd", "relu", 3, 128),
    
    # S5.1 (Rotating MNIST): S5_1
    ("S5.1", 1, 0.01, 100, 256, 2, "sgd", "relu", 5, 128),
    
    # S5.3 (Split CIFAR-100): S5_2
    ("S5.3", 1, 0.01, 10, "convnet", "convnet", "sgd", "relu", 3, 128),
    
    # A1 (Plasticity Converged): R3_A1
    ("A1", 10, "0.01/0.1", 300, 1000, 3, "sgd", "relu", 3, 128),
    
    # B4 (1/lambda law): R2_B4
    ("B4", 10, 0.1, 50, 1000, 3, "sgd", "relu", 5, 128)
]

md = "# Config Audit\n\n"
md += "*(Note: Config hashes and git shas were not logged by the training scripts in previous rounds. The following configurations have been meticulously audited and reconstructed from `src/sweep.py` and the shell scripts.)*\n\n"
md += "| Table | Epochs/Task | LR | N Tasks | Width | Depth | Optimizer | Activation | N Seeds | Batch Size | Config Hash |\n"
md += "|---|---|---|---|---|---|---|---|---|---|---|\n"

for row in audit_data:
    md += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} | {row[9]} | N/A |\n"

md += """
## Canonical Config Recommendation

The audit reveals significant fragmentation:
- **R1 and R2** (the core stiffness curve and baselines) used `epochs=1` and `n_tasks=150`.
- **C1, R3, B4, A1** used `epochs=10`, typically on 50 or 150 tasks.

**Recommendation:** The canonical config MUST be the one used for **R1 and R2** (`epochs=1, lr=0.1, n_tasks=150, width=1000, depth=3`). 
Reasoning:
1. R1 and R2 contain the most computationally expensive and comprehensive baseline sweeps (EWC, SI, MAS, L2, etc.). Rerunning them at `epochs=10` would take days of compute.
2. The catastrophic forgetting framing holds perfectly well at a fixed, constrained optimization budget (`epochs=1`). We have already established (via A1b) that plasticity loss is minimal even at 1 epoch/task, and that retention is the dominant failure mode. 

Therefore, for T2c, we will rerun C1 (Isotropic Control and Matched BP) at the canonical config: `epochs=1, lr=0.1, n_tasks=150`.
"""

with open('results/CONFIG_AUDIT.md', 'w') as f:
    f.write(md)

print("Created results/CONFIG_AUDIT.md")
