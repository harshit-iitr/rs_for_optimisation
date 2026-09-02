"""Schema and code versioning for run artifacts.

SCHEMA_VERSION must be incremented whenever a column is added to, removed from,
or has its meaning changed in metrics.parquet. Columns are never renamed or
repurposed in place -- see experiments/_archive/round1_4_stiffness/results/REPO_AUDIT.md
section 2.1 for what happens when they are (eight undated schema generations).

Round 5 starts at 100 to leave the archived 20/21/22/23/26/28/30/31-column
generations unambiguously in the 0-31 range.
"""

SCHEMA_VERSION = 101  # 100 -> 101: added hessian_trace, sigma_max columns (T9)

# Round-5 canonical experiment configuration. Any study that deviates must say so
# in its STUDY.md. Consumed by src/config.py so it lands in every config.json.
CANONICAL = {
    "dataset": "permuted_mnist",
    "model": "mlp",
    "width": 1000,
    "depth": 3,
    "act_fn": "relu",
    "n_tasks": 150,
    "epochs": 10,
    "lr": 0.1,
    "optimizer": "sgd",
    "batch_size": 256,
    "clip_norm": 0.5,
    "probe_size": 2000,
}
