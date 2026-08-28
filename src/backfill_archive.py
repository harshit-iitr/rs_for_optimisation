"""Reconstruct config.json for the Round 1-4 archive.

No archived run recorded its configuration: run directories contain only
metrics.parquet, and the parquet embeds just method / lambda_rs /
hard_projection / seed. Learning rate, epochs, task count, width, optimizer and
weight decay existed only in the run-id string and in whichever version of
sweep.py was on disk that day (audit section 4.1).

This script reconstructs them from the run-id together with the archived launcher
definitions, and stamps every result "provenance": "reconstructed" so a
reconstructed config is never mistaken for a recorded one. The schema generation
is fingerprinted from the parquet's column set, which is the archive's only
reliable clock (audit section 2.1).

Runs whose configuration cannot be recovered are stamped "unrecoverable" and
listed, rather than being given a plausible guess.
"""

import json
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(REPO, "experiments", "_archive", "round1_4_stiffness", "results")

# Column-set size -> (schema generation, distinguishing column). From audit 2.1.
SCHEMA_BY_NCOLS = {
    20: "gen7_oldest_no_retention", 21: "gen6_method_added",
    22: "gen5_prev_tasks_acc_added", 23: "gen4_update_norm_added",
    26: "gen3_drift_added", 28: "gen2_g_rad_norm_added",
    30: "gen1_drift_rad_tan_added", 31: "gen0_task_0_acc_added",
}

BASE = dict(dataset="permuted_mnist", model="mlp", width=1000, depth=3,
            act_fn="relu", optimizer="sgd", batch_size=256, clip_norm=0.5,
            probe_size=2000, weight_decay=0.0, l2_init_coef=0.01, si_coef=1.0,
            mas_coef=1.0, sp_shrink=0.8, sp_noise=0.01, ewc_lambda=1000.0,
            hard_projection=False, track_drift=False)

# (regex, source launcher, config overrides). First match wins, so order matters.
RULES = [
 (r"^S1_lam_(?P<lam>[\d.]+|inf)_seed", "sweep.py:S1",
  dict(n_tasks=150, epochs=1, lr=0.01)),
 (r"^S2_(?P<arm>.+?)_seed", "sweep.py:S2", dict(n_tasks=150, epochs=1, lr=0.01)),
 (r"^S4_3_width_(?P<w>\d+)_seed", "sweep.py:S4_3",
  dict(n_tasks=150, epochs=1, lr=0.01, method="rs", lambda_rs=1.0)),
 (r"^S5_1_rot_lam_(?P<lam>[\d.]+)_seed", "sweep.py:S5_1",
  dict(dataset="rotating_mnist", n_tasks=100, epochs=1, lr=0.01, width=256, depth=2)),
 (r"^S5_2_(?P<arm>.+?)_seed", "sweep.py:S5_2 / run_s5_cpu.py",
  dict(dataset="split_cifar100", model="convnet", n_tasks=10, epochs=1,
       lr=0.01, probe_size=256)),
 (r"^S5_3_(?P<act>relu|leaky_relu)_lam_(?P<lam>[\d.]+)_seed", "sweep.py:S5_3",
  dict(n_tasks=150, epochs=1, lr=0.01)),
 (r"^S6_1_lr_(?P<lr>[\d.]+)_(?P<arm>.+?)_seed", "sweep.py:S6_1",
  dict(n_tasks=150, epochs=1)),
 (r"^S6_2_opt_(?P<opt>\w+)_seed", "sweep.py:S6_2",
  dict(n_tasks=150, epochs=1, lr=0.01, method="rs", lambda_rs=1.0)),
 (r"^B1_lr_(?P<lr>[\d.]+)_seed", "run_b1_b2.py",
  dict(n_tasks=300, epochs=1, method="bp")),
 (r"^B2_ep_(?P<ep>\d+)_seed", "run_b1_b2.py",
  dict(n_tasks=50, lr=0.01, method="bp")),
 (r"^B4_lam_(?P<lam>[\d.]+)_seed", "run_b4_c1.py",
  dict(n_tasks=50, epochs=10, lr=0.01, method="rs")),
 (r"^C1_A_lr_(?P<lr>[\d.]+)_seed", "run_b4_c1.py",
  dict(n_tasks=150, epochs=10, method="bp")),
 (r"^C1_B_lam_(?P<lam>[\d.]+)_seed", "run_b4_c1.py",
  dict(n_tasks=150, epochs=10, lr=0.1)),
 (r"^R2_B1_lr_(?P<lr>[\d.]+)_seed", "sweep.py:R2_B1",
  dict(n_tasks=300, epochs=1, method="bp")),
 (r"^R2_B2_ep_(?P<ep>\d+)_seed", "sweep.py:R2_B2",
  dict(n_tasks=50, lr=0.01, method="bp")),
 (r"^R2_B4_lam_(?P<lam>[\d.]+)_seed", "sweep.py:R2_B4",
  dict(n_tasks=50, epochs=10, lr=0.1, method="rs")),
 (r"^R2_C1_A_bp_lr_(?P<lr>[\d.]+)_seed", "sweep.py:R2_C1",
  dict(n_tasks=50, epochs=10, method="bp")),
 (r"^R2_C1_B_rs_lam_(?P<lam>[\d.]+)_seed", "sweep.py:R2_C1",
  dict(n_tasks=50, epochs=10, lr=0.1)),
 (r"^R3_R1_lam_(?P<lam>[\d.]+|inf)_seed", "sweep.py:R3_R1",
  dict(n_tasks=150, epochs=10, lr=0.1, track_drift=True)),
 (r"^R3_R2_(?P<arm>.+?)_seed", "sweep.py:R3_R2", dict(n_tasks=150, epochs=10, lr=0.1)),
 (r"^R3_R4_w_(?P<w>\d+)_lam_(?P<lam>[\d.]+)_seed", "sweep.py:R3_R4",
  dict(n_tasks=150, epochs=10, lr=0.1)),
 (r"^R3_R6_opt_(?P<opt>sgd_momentum|sgd|adamw|adam)_lam_(?P<lam>[\d.]+)_seed",
  "sweep.py:R3_R6", dict(n_tasks=150, epochs=10)),
 (r"^R3_A1b_ep_(?P<ep>\d+)_seed", "sweep.py:R3_A1b",
  dict(n_tasks=300, lr=0.01, method="bp")),
 (r"^R3_A1_lr_(?P<lr>[\d.]+)_seed", "sweep.py:R3_A1",
  dict(n_tasks=300, epochs=10, method="bp")),
 (r"^R3_A2_lam_(?P<lam>[\d.]+)_seed", "sweep.py:R3_A2",
  dict(n_tasks=50, epochs=10, lr=0.1, method="rs")),
 (r"^R3_C1_bp_iso_seed", "sweep.py:R3_C1_iso",
  dict(n_tasks=50, epochs=10, lr=0.1, method="bp_iso")),
 (r"^R3_C1_bp_matched_lr_(?P<lr>[\d.]+)_seed", "sweep.py:R3_C1_matched",
  dict(n_tasks=50, epochs=10, method="bp")),
 (r"^R4_T1_lam_(?P<lam>[\d.]+)_seed", "sweep.py:R4_T1",
  dict(n_tasks=150, epochs=10, lr=0.1, track_drift=True)),
 (r"^R4_T2c_(?P<arm>bp_iso|bp|rs)_seed", "sweep.py:R4_T2c",
  dict(n_tasks=150, epochs=10, lr=0.1, track_drift=True)),
 (r"^R4_T3a_rot_lam_(?P<lam>[\d.]+)_seed", "sweep.py:R4_T3a",
  dict(dataset="rotating_mnist", n_tasks=100, epochs=1, lr=0.01, width=256, depth=2)),
 (r"^R4_T3b_(?P<arm>.+?)_seed", "sweep.py:R4_T3b",
  dict(dataset="split_cifar100", model="convnet", n_tasks=10, epochs=1,
       lr=0.01, probe_size=256)),
]

ARM_METHOD = {
    "bp": ("bp", 0.0), "cbp": ("cbp", 0.0), "redo": ("redo", 0.0),
    "er": ("er", 0.0), "sp": ("sp", 0.0), "ln": ("ln", 0.0),
    "l2_init": ("l2_init", 0.0), "si": ("si", 0.0), "mas": ("mas", 0.0),
    "rs": ("rs", None), "rs_cbp": ("rs_cbp", 1.0),
    "l2_1e-3": ("l2", 0.0), "l2_1e-4": ("l2", 0.0), "l2": ("l2", 0.0),
    "ln_l2": ("ln_l2", 0.0), "l2_er": ("l2_er", 0.0),
    "bp_iso": ("bp_iso", 0.0), "ewc": ("ewc", 0.0),
}

# Known-bad arms. Recorded on the config so no future analysis can pick them up
# without seeing why they are untrustworthy.
DEFECTS = {
    "cbp": "INERT: CBP was never effective (replacement_rate 1e-6, uniform-random "
           "selection ignoring utilities) and is not instantiated by the current "
           "trainer at all. Indistinguishable from bp. Audit 5.1.",
    "rs_cbp": "Contains the inert CBP component. Audit 5.1.",
    "er": "ER penalty is an unnormalized Frobenius norm at a hardcoded 0.01 "
          "coefficient; collapses the network to ~0.11 accuracy. Audit 5.5a.",
    "l2_er": "See 'er'. Audit 5.5a.",
    "mas": "Omega accumulates without decay; the network never trains "
           "(test acc 0.41), so its retention is an artifact. Audit 5.5b.",
    "bp_iso": "INVALID CONTROL: gradient scaling applied before clip_grad_norm_, "
              "which renormalized it away. Achieved 0.7% of an intended 18.5% "
              "match. Audit 3.3.2.",
}


def reconstruct(run_id):
    for pat, src, over in RULES:
        m = re.match(pat, run_id)
        if not m:
            continue
        g = m.groupdict()
        cfg = dict(BASE); cfg.update(over)
        cfg["reconstructed_from"] = src
        if "lr" in g and g["lr"]:
            cfg["lr"] = float(g["lr"])
        if "ep" in g and g["ep"]:
            cfg["epochs"] = int(g["ep"])
        if "w" in g and g["w"]:
            cfg["width"] = int(g["w"])
        if "act" in g and g["act"]:
            cfg["act_fn"] = g["act"]
        if "opt" in g and g["opt"]:
            cfg["optimizer"] = g["opt"]
            if src == "sweep.py:R3_R6":
                # The bug that made adam and adamw bitwise identical: 'adam' is a
                # substring of 'adamw', so both got lr=0.001; and weight_decay was
                # forced to 0 for bp/rs, making AdamW(0) == Adam(0). Audit 5.3.
                cfg["lr"] = 0.001 if "adam" in g["opt"] else 0.01
        if "lam" in g and g["lam"]:
            if g["lam"] == "inf":
                cfg["hard_projection"] = True
                cfg["method"], cfg["lambda_rs"] = "bp", 0.0
                cfg["projection_note"] = ("straight-through estimator, NOT the "
                                          "constraint limit. Audit 5.5c.")
            else:
                lam = float(g["lam"])
                cfg["lambda_rs"] = lam
                cfg.setdefault("method", "rs" if lam > 0 else "bp")
                if lam == 0:
                    cfg["method"] = "bp"
        if "arm" in g and g["arm"]:
            arm = g["arm"]
            meth, lam = ARM_METHOD.get(arm, (arm, 0.0))
            cfg["method"] = meth
            if lam is not None:
                cfg["lambda_rs"] = lam
            if arm == "l2_1e-3":
                cfg["weight_decay"] = 1e-3
            elif arm in ("l2_1e-4", "ln_l2", "l2_er"):
                cfg["weight_decay"] = 1e-4
            elif arm == "l2" and src == "sweep.py:R3_R2":
                cfg["weight_decay"] = 1e-3
            em = re.match(r"^(rs_)?ewc_(\d+)$", arm)
            if em:
                cfg["method"] = "rs_ewc" if em.group(1) else "ewc"
                cfg["ewc_lambda"] = float(em.group(2))
                cfg["lambda_rs"] = 0.03 if em.group(1) else 0.0
            if arm == "rs" and src in ("sweep.py:R3_R2", "sweep.py:R4_T2c"):
                cfg["lambda_rs"] = 0.03 if src == "sweep.py:R3_R2" else 0.01
            if arm == "rs" and src == "sweep.py:S2":
                cfg["lambda_rs"] = 1.0
            if arm == "rs" and src in ("sweep.py:S5_2 / run_s5_cpu.py",):
                cfg["lambda_rs"] = 1.0
            if arm == "rs" and src == "sweep.py:R4_T3b":
                cfg["lambda_rs"] = 0.01
        cfg.setdefault("method", "bp"); cfg.setdefault("lambda_rs", 0.0)
        cfg.setdefault("epochs", 1); cfg.setdefault("lr", 0.01)
        return cfg
    return None


def main(write=True):
    if not os.path.isdir(ARCHIVE):
        sys.exit(f"archive not found: {ARCHIVE}")
    done = unrec = noparq = 0
    unrecoverable = []
    for run_id in sorted(os.listdir(ARCHIVE)):
        d = os.path.join(ARCHIVE, run_id)
        if not os.path.isdir(d):
            continue
        m = re.search(r"_seed(\d+)$", run_id)
        cfg = reconstruct(run_id)
        pq = os.path.join(d, "metrics.parquet")

        if cfg is None:
            unrec += 1; unrecoverable.append(run_id)
            cfg = {"provenance": "unrecoverable",
                   "note": "No archived launcher defines this run-id. Its "
                           "configuration cannot be recovered; do not use it."}
        else:
            cfg["provenance"] = "reconstructed"
            cfg["reconstruction_warning"] = (
                "Reconstructed from the run-id and the archived launcher "
                "definitions, NOT recorded at run time. Round 1-4 wrote no config.")
            cfg["seed"] = int(m.group(1)) if m else None
            for arm_key, why in DEFECTS.items():
                if cfg.get("method") == arm_key:
                    cfg["KNOWN_DEFECT"] = why

        if os.path.exists(pq):
            try:
                cols = pd.read_parquet(pq).columns
                cfg["n_parquet_columns"] = len(cols)
                cfg["schema_generation"] = SCHEMA_BY_NCOLS.get(
                    len(cols), f"unknown_{len(cols)}col")
                cfg["has_retention_column"] = "prev_tasks_acc" in cols
            except Exception as e:
                cfg["parquet_error"] = str(e)
        else:
            noparq += 1
            cfg["status"] = "failed"
            cfg["note_failure"] = ("No metrics.parquet. This run died before "
                                   "writing output -- almost certainly CUDA OOM "
                                   "from launcher over-concurrency. Audit 4.4.")
        cfg.setdefault("status", "complete")
        cfg["archived_run_id"] = run_id
        if write:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(cfg, f, indent=2)
        done += 1

    print(f"backfilled {done} archived run directories")
    print(f"  {noparq} have no metrics.parquet (failed runs, now stamped as such)")
    print(f"  {unrec} unrecoverable: {unrecoverable}")


if __name__ == "__main__":
    main()
