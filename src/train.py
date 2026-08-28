"""Round 5 trainer.

Rewritten from the Round 1-4 trainer per audit section 7. Every measurement has
moved out of this loop and behind a tested function in src/metrics/. What remains
here is the training step, the probe schedule, and the assembly of the log row.

Corrections carried in from the audit:
  * gradient scaling is applied AFTER clipping, and the ordering is asserted
  * prev_only_acc is logged alongside the (renamed) avg_seen_acc
  * absolute drift is logged alongside relative drift
  * drift against a fixed task-0 reference is logged alongside consecutive drift
  * readiness is per layer, and an assertion catches the broadcast regression
  * the hard-constraint arm is a true tangential projection; the straight-through
    variant is a separate, explicitly-named arm
  * config.json is written BEFORE training starts
  * nothing is ever averaged across layers in here
"""

import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import build_config, file_sha256, finalize_config, write_config
from src.data.permuted_mnist import PermutedMNIST
from src.data.rotating_mnist import RotatingMNIST
from src.methods.baselines import apply_shrink_and_perturb, compute_l2_init_penalty
from src.methods.ewc import EWC
from src.methods.isotropic import GradTrace, IsotropicControl
from src.methods.mas import MAS
from src.methods.rs import compute_rs_penalty
from src.methods.si import SI
from src.metrics.drift import compute_drift, compute_subspace_overlap
from src.metrics.gradients import group_grad_norms, param_groups
from src.metrics.neurons import compute_dead_fraction, compute_dormant_fraction
from src.metrics.norms import compute_activation_radius, compute_grad_norm, compute_weight_norm
from src.metrics.phi_rad import compute_phi_rad_tilde
from src.metrics.rank import compute_effective_rank, compute_stable_rank
from src.metrics.readiness import compute_readiness
from src.metrics.retention import evaluate_tasks, summarize
from src.models.mlp import MLP

METHODS = ["bp", "rs", "isotropic", "l2", "l2_init", "ln", "ln_l2", "sp",
           "ewc", "si", "mas", "rs_ewc"]
PROJECTIONS = ["none", "tangential", "ste"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="bp", choices=METHODS)
    p.add_argument("--lambda_rs", type=float, default=0.0)
    p.add_argument("--projection", type=str, default="none", choices=PROJECTIONS,
                   help="hard-constraint arm: 'tangential' is the true limit, "
                        "'ste' is the Round 1-4 straight-through variant")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--l2_init_coef", type=float, default=0.01)
    p.add_argument("--ewc_lambda", type=float, default=1000.0)
    p.add_argument("--si_coef", type=float, default=1.0)
    p.add_argument("--mas_coef", type=float, default=1.0)
    p.add_argument("--sp_shrink", type=float, default=0.8)
    p.add_argument("--sp_noise", type=float, default=0.01)

    p.add_argument("--dataset", type=str, default="permuted_mnist",
                   choices=["permuted_mnist", "rotating_mnist"])
    p.add_argument("--n_tasks", type=int, default=150)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--width", type=int, default=1000)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--act_fn", type=str, default="relu", choices=["relu", "leaky_relu"])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--optimizer", type=str, default="sgd",
                   choices=["sgd", "sgd_momentum", "adam", "adamw"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--clip_norm", type=float, default=0.5,
                   help="global grad-norm clip; <=0 disables clipping entirely")
    p.add_argument("--probe_size", type=int, default=2000)
    p.add_argument("--drift_probe_size", type=int, default=1000)
    p.add_argument("--readiness_microbatches", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--track_drift", action="store_true")
    p.add_argument("--log_grad_trace", action="store_true",
                   help="record realized per-step gradient magnitudes "
                        "(required for any run used as an isotropic target)")
    p.add_argument("--iso_target", type=str, default=None,
                   help="path to the target run's grad_trace.npz (method=isotropic)")
    p.add_argument("--iso_granularity", type=str, default="per_layer",
                   choices=["per_layer", "global"])
    return p.parse_args()


def main():
    args = build_args()
    t_start = time.time()

    if args.method == "isotropic" and not args.iso_target:
        raise SystemExit("--method isotropic requires --iso_target")

    # Config sidecar FIRST, so a run that dies mid-way is still identifiable.
    os.makedirs(args.run_dir, exist_ok=True)
    cfg = build_config(args, extra={"iso_target_sha256": file_sha256(args.iso_target)})
    write_config(args.run_dir, cfg)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    act_class = nn.ReLU if args.act_fn == "relu" else nn.LeakyReLU
    use_ln = args.method in ("ln", "ln_l2")

    model = MLP(use_ln=use_ln, depth=args.depth, width=args.width,
                act_fn=act_class).to(device)
    groups = param_groups(model)
    group_names = [n for n, _ in groups]

    initial_model = None
    if args.method == "l2_init":
        initial_model = copy.deepcopy(model).eval()
    ewc = EWC(model) if args.method in ("ewc", "rs_ewc") else None
    si = SI(model) if args.method == "si" else None
    mas = MAS(model) if args.method == "mas" else None

    iso = None
    if args.method == "isotropic":
        iso = IsotropicControl(args.iso_target, args.iso_granularity)
        iso.check_compatible(group_names)

    trace = GradTrace(group_names) if (args.log_grad_trace or iso is not None) else None

    ds_cls = {"permuted_mnist": PermutedMNIST, "rotating_mnist": RotatingMNIST}[args.dataset]
    dataset = ds_cls(n_tasks=args.n_tasks, device=device, seed=args.seed)

    wd = args.weight_decay if args.method in ("l2", "ln_l2") else 0.0
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=wd)
    elif args.optimizer == "sgd_momentum":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=wd)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    fwd = {} if args.projection == "none" else {"projection": args.projection}
    records, task_tests = [], []
    global_step = 0

    if args.track_drift:
        probe0, _, _, _ = dataset.get_task_data(0)
        drift_probe_x = probe0[:args.drift_probe_size].clone().to(device)
        past_acts, ref_acts = None, None

    for t in tqdm(range(args.n_tasks), desc=os.path.basename(args.run_dir)):
        x_train, y_train, x_test, y_test = dataset.get_task_data(t)
        task_tests.append((x_test, y_test))

        model.eval()
        with torch.no_grad():
            init_logits = model(x_train[:args.probe_size], **fwd)
            init_acc = (init_logits.argmax(1) == y_train[:args.probe_size]).float().mean().item()

        n_samples = x_train.shape[0]
        model.train()
        last_update_norms = {l: 0.0 for l in range(len(model.layers))}

        for ep in range(args.epochs):
            indices = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, args.batch_size):
                bidx = indices[start:start + args.batch_size]
                optimizer.zero_grad(set_to_none=False)
                out, pre_acts, _ = model(x_train[bidx], return_activations=True, **fwd)
                loss = criterion(out, y_train[bidx])

                if args.method == "l2_init":
                    loss = loss + args.l2_init_coef * compute_l2_init_penalty(model, initial_model)
                if ewc is not None:
                    loss = loss + args.ewc_lambda * ewc.penalty()
                if si is not None:
                    loss = loss + args.si_coef * si.penalty()
                if mas is not None:
                    loss = loss + args.mas_coef * mas.penalty()
                if args.lambda_rs > 0 and args.projection == "none":
                    loss = loss + args.lambda_rs * sum(compute_rs_penalty(h) for h in pre_acts)

                loss.backward()

                # ---- gradient post-processing: ORDER IS LOAD-BEARING ----
                # 1. clip   2. isotropic scaling
                # Round 1-4 had these reversed, so clip_grad_norm_ renormalized the
                # scaling away and the control silently did nothing.
                _, pre_clip_global = group_grad_norms(groups)
                if args.clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                post_clip_norms, post_clip_global = group_grad_norms(groups)

                if iso is not None:
                    realized = iso.apply(groups, global_step)
                    # The invariant that the whole control rests on: after
                    # scaling, the realized magnitude IS the target. If clipping
                    # ran afterwards -- the Round 1-4 ordering -- this fails.
                    tgt = iso.target_global(global_step)
                    if tgt is not None and tgt > 1e-9 and global_step % 97 == 0:
                        got = float(np.linalg.norm(realized))
                        assert abs(math.log(max(got, 1e-12) / tgt)) < 1e-3, (
                            f"isotropic magnitude match broken at step {global_step}: "
                            f"realized {got:.6g} vs target {tgt:.6g} -- check that "
                            f"clipping does not run after the scaling")
                else:
                    realized = post_clip_norms

                if trace is not None:
                    trace.record(t, pre_clip_global, realized)

                is_last = (ep == args.epochs - 1) and (start + args.batch_size >= n_samples)
                if is_last:
                    old_w = [layer.weight.clone().detach() for layer in model.layers]
                if si is not None:
                    old_params = {n: p.data.clone() for n, p in model.named_parameters()
                                  if p.requires_grad}

                optimizer.step()

                if si is not None:
                    si.update_W(old_params)
                if is_last:
                    for l, layer in enumerate(model.layers):
                        last_update_norms[l] = torch.norm(
                            layer.weight.detach() - old_w[l]).item()
                global_step += 1

        if args.method == "sp":
            apply_shrink_and_perturb(model, shrink=args.sp_shrink, noise_sigma=args.sp_noise)
        if ewc is not None:
            model.zero_grad()
            criterion(model(x_train[:args.probe_size], **fwd),
                      y_train[:args.probe_size]).backward()
            ewc.update_fisher(); model.zero_grad()
        if si is not None:
            si.update_omega()
        if mas is not None:
            model.zero_grad()
            model(x_train[:args.probe_size], **fwd).pow(2).sum(1).mean().backward()
            mas.update_omega(); model.zero_grad()

        # ---------------- end-of-task probes ----------------
        n_probe = min(args.probe_size, len(x_test))
        probe_x, probe_y = x_test[:n_probe], y_test[:n_probe]

        # Readiness, PER LAYER. Round 1-4 computed one network-wide value and
        # broadcast it into every layer row (audit section 5.2).
        m = args.readiness_microbatches
        micro = n_probe // m
        per_layer_micro = {l: [] for l in range(len(model.layers))}
        for i in range(m):
            mb_x, mb_y = probe_x[i * micro:(i + 1) * micro], probe_y[i * micro:(i + 1) * micro]
            if len(mb_x) == 0:
                continue
            model.zero_grad()
            criterion(model(mb_x, **fwd), mb_y).backward()
            for l, layer in enumerate(model.layers):
                if layer.weight.grad is not None:
                    per_layer_micro[l].append(layer.weight.grad.detach().clone().view(-1))
        readiness = {l: (compute_readiness(v) if v else float("nan"))
                     for l, v in per_layer_micro.items()}
        if len(readiness) > 1:
            vals = [v for v in readiness.values() if not math.isnan(v)]
            assert len(set(vals)) > 1 or len(vals) <= 1, \
                "readiness identical across layers -- broadcast regression"

        # phi_rad uses the TASK-LOSS gradient only. The probe loss never contains
        # the penalty; the penalty gradient is purely radial by construction, so
        # including it would make the metric self-referential. Regression test:
        # tests/test_phi_rad.py::test_phi_rad_tilde_leakage
        model.zero_grad()
        out_p, pre_p, post_p = model(probe_x, return_activations=True, **fwd)
        loss_p = criterion(out_p, probe_y)
        g_tasks = torch.autograd.grad(loss_p, pre_p, retain_graph=True)
        loss_p.backward()

        layer_metrics = []
        for l in range(len(pre_p)):
            h, h_post, g = pre_p[l].detach(), post_p[l].detach(), g_tasks[l].detach()
            h_hat = h / torch.norm(h, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            g_rad = (g * h_hat).sum(-1, keepdim=True) * h_hat
            rad_mean, rad_std = compute_activation_radius(h)
            layer_metrics.append({
                "layer": l,
                "phi_rad_tilde": compute_phi_rad_tilde(h, g),
                "radial_excess": rad_mean - math.sqrt(h.shape[-1]),
                "radius_mean": rad_mean, "radius_std": rad_std,
                "eff_rank": compute_effective_rank(h),
                "stable_rank": compute_stable_rank(h),
                "dead_frac": compute_dead_fraction(h_post),
                "dormant_frac": compute_dormant_fraction(h_post),
                "weight_norm": compute_weight_norm(model.layers[l].weight),
                "grad_norm": compute_grad_norm(model.layers[l].weight.grad),
                "update_norm": last_update_norms[l],
                "g_rad_norm": torch.sqrt((g_rad ** 2).sum(-1)).mean().item(),
                "g_norm_task": torch.norm(g, p=2, dim=-1).mean().item(),
                "readiness": readiness.get(l, float("nan")),
            })

        drift_keys = ["drift_abs", "drift_rad_abs", "drift_tan_abs", "ref_norm",
                      "drift_rel", "drift_rad", "drift_tan", "drift_cos_sim"]
        sub_keys = ["subspace_overlap", "subspace_proj_metric", "subspace_k"]
        for lm in layer_metrics:
            for k in drift_keys + sub_keys:
                lm[k] = float("nan")
                lm[k + "_ref"] = float("nan")

        if args.track_drift:
            model.eval()
            with torch.no_grad():
                _, curr_acts, _ = model(drift_probe_x, return_activations=True, **fwd)
            curr_acts = [h.detach() for h in curr_acts]
            for l, h_curr in enumerate(curr_acts):
                lm = layer_metrics[l]
                if past_acts is not None:      # consecutive task boundaries
                    for k, v in compute_drift(h_curr, past_acts[l]).items():
                        lm[k] = v
                    for k, v in compute_subspace_overlap(h_curr, past_acts[l]).items():
                        lm[k] = v
                if ref_acts is not None:       # fixed task-0 reference
                    for k, v in compute_drift(h_curr, ref_acts[l]).items():
                        lm[k + "_ref"] = v
                    for k, v in compute_subspace_overlap(h_curr, ref_acts[l]).items():
                        lm[k + "_ref"] = v
            past_acts = [h.clone() for h in curr_acts]
            if ref_acts is None:
                ref_acts = [h.clone() for h in curr_acts]

        accs = evaluate_tasks(model, task_tests, args.probe_size, fwd)
        ret = summarize(accs)
        model.eval()
        with torch.no_grad():
            test_acc = accs[-1]
            train_acc = (model(x_train[:args.probe_size], **fwd).argmax(1)
                         == y_train[:args.probe_size]).float().mean().item()
        model.train()

        common = dict(
            task=t, method=args.method, lambda_rs=args.lambda_rs,
            projection=args.projection, seed=args.seed,
            schema_version=cfg["schema_version"], config_hash=cfg["config_hash"],
            init_acc=init_acc, train_acc=train_acc, test_acc=test_acc,
            first_epoch_gain=train_acc - init_acc, **ret,
        )
        for lm in layer_metrics:
            records.append({**common, **lm})

    df = pd.DataFrame(records)
    df.to_parquet(os.path.join(args.run_dir, "metrics.parquet"))
    if trace is not None:
        trace.save(os.path.join(args.run_dir, "grad_trace.npz"))

    extra = {"n_rows": len(df), "n_steps": global_step}
    if iso is not None:
        extra["iso_target_exhausted"] = bool(iso.exhausted)
    finalize_config(args.run_dir, "complete", time.time() - t_start, extra)


if __name__ == "__main__":
    main()
