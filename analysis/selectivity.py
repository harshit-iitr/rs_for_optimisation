"""Per-neuron task selectivity from an end-of-run checkpoint.

Answers: are hidden units task-shared or task-specific, and does the radial
penalty move the network between those regimes?

For each layer separately -- per-layer metrics are never averaged across layers
(README, convention 3) -- and for each hidden unit i and task k:

    resp[i, k] = mean over task k's probe inputs of |post-activation_i|
    sel[i, k]  = resp[i, k] / max_k resp[i, k]

The per-unit normalisation is the point: an unnormalised map shows which units
are loud, not which units are selective.

Units dead on every task (max response exactly 0) cannot be normalised. They are
dropped and counted rather than silently becoming NaN rows.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.mlp import MLP
from src.data.permuted_mnist import PermutedMNIST
from src.data.rotating_mnist import RotatingMNIST


def load_run(run_dir, device="cuda"):
    ckpt_path = os.path.join(run_dir, "final_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{run_dir} has no final_model.pt -- it was run without "
            f"--save_final_checkpoint. Re-run that arm; the checkpoint cannot be "
            f"reconstructed after the fact.")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    if cfg.get("status") != "complete":
        raise ValueError(f"{run_dir} status={cfg.get('status')}, not reportable")
    act = nn.ReLU if ck["act_fn"] == "relu" else nn.LeakyReLU
    model = MLP(use_ln=ck["use_ln"], depth=ck["depth"], width=ck["width"],
                act_fn=act).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck, cfg


def responses(run_dir, probe_size=2000, tasks=None, device="cuda"):
    """(n_layers, units, n_tasks) mean absolute post-activation response."""
    model, ck, cfg = load_run(run_dir, device)
    ds_cls = {"permuted_mnist": PermutedMNIST,
              "rotating_mnist": RotatingMNIST}[ck["dataset"]]
    # Same seed as the run, so the task sequence is the one it actually saw.
    dataset = ds_cls(n_tasks=ck["n_tasks"], device=device, seed=ck["seed"])
    task_ids = list(range(ck["n_tasks"])) if tasks is None else list(tasks)
    fwd = {} if ck["projection"] in (None, "none") else {"projection": ck["projection"]}

    cols = []
    with torch.no_grad():
        for t in task_ids:
            _, _, x_test, _ = dataset.get_task_data(t)
            x = x_test[:probe_size].to(device)
            _, _, post = model(x, return_activations=True, **fwd)
            cols.append(np.stack([p.abs().mean(0).float().cpu().numpy() for p in post]))
    return np.stack(cols, axis=-1), ck, cfg


def selectivity(resp):
    """Per-unit-normalised map, index, participation ratio, dead count.

    resp: (units, n_tasks) for ONE layer."""
    mx = resp.max(axis=1)
    alive = mx > 0
    n_dead = int((~alive).sum())
    sel = resp[alive] / mx[alive][:, None]
    index = 1.0 - sel.mean(axis=1)
    s1, s2 = sel.sum(axis=1), (sel ** 2).sum(axis=1)
    pr = (s1 ** 2) / np.maximum(s2, 1e-12)
    return sel, index, pr, n_dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--probe_size", type=int, default=2000)
    ap.add_argument("--every", type=int, default=1,
                    help="use every Nth task; 150 columns render fine, so the "
                         "default keeps all of them")
    ap.add_argument("--out", default=None, help="write the .npz here")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_tasks = json.load(open(os.path.join(a.run_dir, "config.json")))["n_tasks"]
    tasks = list(range(0, n_tasks, a.every))
    resp, ck, cfg = responses(a.run_dir, a.probe_size, tasks, device)

    out = {"tasks": np.array(tasks), "config_hash": cfg["config_hash"]}
    print(f"{a.run_dir}\n  config_hash {cfg['config_hash']}  seed {ck['seed']}  "
          f"{resp.shape[0]} layers x {resp.shape[1]} units x {resp.shape[2]} tasks")
    for L in range(resp.shape[0]):
        sel, index, pr, n_dead = selectivity(resp[L])
        out[f"resp_layer{L}"] = resp[L]
        out[f"sel_layer{L}"] = sel
        out[f"index_layer{L}"] = index
        out[f"pr_layer{L}"] = pr
        print(f"  layer {L}: selectivity_index {index.mean():.4f} +/- {index.std():.4f} | "
              f"participation_ratio {pr.mean():.2f} of {resp.shape[2]} tasks | "
              f"dead {n_dead}/{resp.shape[1]}")
    path = a.out or os.path.join(a.run_dir, "selectivity.npz")
    np.savez_compressed(path, **out)
    print("  ->", path)


if __name__ == "__main__":
    main()
