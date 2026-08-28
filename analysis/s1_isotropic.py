"""S1 -- the isotropic control. The paper's decisive experiment.

Order of operations is deliberate and is the point of the whole script:

  1. Report the SIZE OF THE MAGNITUDE CONFOUND: how far apart the baseline and
     penalty arms actually are in realized gradient magnitude. If it is ~0 there
     is nothing for a magnitude-matched control to remove, and any null result
     from that control means nothing.
  2. Run the ACCEPTANCE TEST on each control arm. An arm that fails is not
     reported (rule 7).
  3. Only then, the retention comparison.

Round 1-4 did step 3 alone, on a control that had silently achieved 4% of its
intended match, and reported the resulting null as proof of directionality.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.common import (EXP, IncompleteArm, config_header, fmt_paired,
                             load_arm, outdir, paired, per_seed)
from src.methods.isotropic import acceptance_test
from src.studies import LAMBDA_STAR, STUDIES

STUDY = "S1_isotropic_control"
ROOT = os.path.join(EXP, STUDIES[STUDY]["root"])
SEEDS = STUDIES[STUDY]["seeds"]
PENALTY = f"arm_penalty_lam{LAMBDA_STAR}"
ARMS = ["arm_baseline", PENALTY, "arm_isotropic_per_layer", "arm_isotropic_global"]
METRICS = ["prev_only_acc", "avg_seen_acc", "test_acc", "task_0_acc",
           "weight_norm", "radius_mean", "drift_abs", "drift_rel",
           "subspace_overlap"]


def confound_size(regime):
    """How far apart are baseline and penalty in realized gradient magnitude?"""
    rows = []
    for s in SEEDS:
        b = os.path.join(ROOT, regime, "arm_baseline", f"seed_{s}", "grad_trace.npz")
        p = os.path.join(ROOT, regime, PENALTY, f"seed_{s}", "grad_trace.npz")
        if not (os.path.exists(b) and os.path.exists(p)):
            continue
        B, P = np.load(b, allow_pickle=True), np.load(p, allow_pickle=True)
        names = [str(x) for x in P["group_names"]]
        bn, pn = B["post"], P["post"]
        n = min(len(bn), len(pn))
        r = {"seed": s,
             "global": float(np.median(np.linalg.norm(pn[:n], axis=1)
                                       / np.linalg.norm(bn[:n], axis=1))),
             "clip_binds_frac": float(np.mean(B["pre_global"][:n] > 0.0)
                                      if regime == "unclipped" else
                                      np.mean(B["pre_global"][:n] > 0.5))}
        for i, nm in enumerate(names):
            m = (bn[:n, i] > 1e-12) & (pn[:n, i] > 1e-12)
            r[nm] = float(np.median(pn[:n][m, i] / bn[:n][m, i])) if m.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def gate(regime, arm, granularity):
    reports = {}
    for s in SEEDS:
        real = os.path.join(ROOT, regime, arm, f"seed_{s}", "grad_trace.npz")
        tgt = os.path.join(ROOT, regime, PENALTY, f"seed_{s}", "grad_trace.npz")
        if not (os.path.exists(real) and os.path.exists(tgt)):
            reports[s] = {"passed": False, "reason": "missing grad_trace"}
            continue
        ok, rep = acceptance_test(real, tgt, granularity, tol=0.05)
        reports[s] = rep
    passed = all(r.get("passed") for r in reports.values())
    return passed, reports


def main():
    out = outdir(STUDY)
    summary = {"study": STUDY, "lambda_star": LAMBDA_STAR, "regimes": {}}
    print("=" * 78)
    print("S1 -- ISOTROPIC CONTROL")
    print("=" * 78)

    for regime in ["clipped", "unclipped"]:
        print(f"\n\n{'#' * 78}\n# REGIME: {regime}\n{'#' * 78}")
        reg = {}
        arms, cfg0 = {}, None
        for a in ARMS:
            try:
                d = load_arm(os.path.join(ROOT, regime, a), SEEDS)
                arms[a] = d
                cfg0 = cfg0 or d.attrs["config"]
            except (IncompleteArm, FileNotFoundError) as e:
                print(f"  [arm not reportable] {a}: {e}")
        if not arms:
            print("  no complete arms yet"); continue
        print(f"\nconfig: {config_header(cfg0)}")

        # ---- 1. the confound -------------------------------------------------
        print(f"\n--- 1. MAGNITUDE CONFOUND (penalty / baseline, median over steps) ---")
        cs = confound_size(regime)
        if not cs.empty:
            m = cs.drop(columns=["seed"]).mean()
            print(cs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print(f"\n  mean over seeds: " +
                  "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
            reg["confound"] = {"per_seed": cs.to_dict("records"), "mean": m.to_dict()}
            gmean = float(m.get("global", np.nan))
            print(f"\n  => global magnitude confound: {abs(1 - gmean) * 100:.2f}%")
            grp = {k: v for k, v in m.items()
                   if k not in ("seed", "global", "clip_binds_frac")}
            worst = max(abs(1 - v) for v in grp.values()) if grp else float("nan")
            print(f"  => per-layer magnitude confound: up to {worst * 100:.1f}% "
                  f"({min(grp, key=lambda k: -abs(1 - grp[k]))})")
            if abs(1 - gmean) < 0.005:
                print("\n     The GLOBAL confound is negligible: clipping pins both "
                      "arms to the same total\n     gradient norm, so a global-norm "
                      "control matches something already equal and\n     its null is "
                      "uninformative. The PER-LAYER confound is real, so the\n     "
                      "per-layer arm does have something to remove and its result "
                      "does carry\n     information.")

        # ---- 2. the gate -----------------------------------------------------
        print(f"\n--- 2. ACCEPTANCE TEST (pre-registered: median |log ratio| < 0.05) ---")
        reportable = ["arm_baseline", PENALTY]
        reg["acceptance"] = {}
        for arm, gran in [("arm_isotropic_per_layer", "per_layer"),
                          ("arm_isotropic_global", "global")]:
            if arm not in arms:
                continue
            ok, reps = gate(regime, arm, gran)
            vals = [r.get("criterion_value") for r in reps.values() if "criterion_value" in r]
            grp = [r.get("worst_per_group_median_abs_log_ratio") for r in reps.values()
                   if "worst_per_group_median_abs_log_ratio" in r]
            print(f"  {arm:28s} granularity={gran:9s} PASSED={ok}  "
                  f"criterion(max over seeds)={max(vals) if vals else float('nan'):.2e}  "
                  f"worst per-group residual={max(grp) if grp else float('nan'):.4f}")
            reg["acceptance"][arm] = {"passed": bool(ok),
                                      "criterion_max": float(max(vals)) if vals else None,
                                      "worst_group_residual": float(max(grp)) if grp else None}
            if ok:
                reportable.append(arm)
            else:
                print(f"     -> NOT REPORTED (rule 7).")

        # ---- 3. the comparison ----------------------------------------------
        print(f"\n--- 3. OUTCOMES (final 20 tasks, layer 0, n={len(SEEDS)} seeds) ---")
        stats_by_arm = {a: per_seed(arms[a], METRICS) for a in reportable if a in arms}
        tbl = pd.DataFrame({a: s.mean() for a, s in stats_by_arm.items()}).T
        sd = pd.DataFrame({a: s.std(ddof=1) for a, s in stats_by_arm.items()}).T
        show = pd.DataFrame(index=tbl.index)
        for c in ["test_acc", "prev_only_acc", "avg_seen_acc", "task_0_acc",
                  "weight_norm", "radius_mean"]:
            if c in tbl:
                show[c] = [f"{tbl.loc[i, c]:.4f}+-{sd.loc[i, c]:.4f}" for i in tbl.index]
        print(show.to_string())
        reg["outcomes"] = {a: {"mean": s.mean().to_dict(), "sd": s.std(ddof=1).to_dict()}
                           for a, s in stats_by_arm.items()}

        print(f"\n--- paired comparisons on prev_only_acc ---")
        reg["tests"] = []
        base = stats_by_arm.get("arm_baseline")
        pen = stats_by_arm.get(PENALTY)
        for metric in ["prev_only_acc", "test_acc"]:
            print(f"  [{metric}]")
            for a, b, la, lb in [
                ("arm_baseline", PENALTY, "baseline", "penalty"),
                ("arm_baseline", "arm_isotropic_per_layer", "baseline", "iso_per_layer"),
                ("arm_baseline", "arm_isotropic_global", "baseline", "iso_global"),
                (PENALTY, "arm_isotropic_per_layer", "penalty", "iso_per_layer"),
            ]:
                if a in stats_by_arm and b in stats_by_arm:
                    r = paired(stats_by_arm[a], stats_by_arm[b], la, lb, metric)
                    print(fmt_paired(r)); reg["tests"].append(r)

        # fraction of the penalty's gain that magnitude matching recovers
        if base is not None and pen is not None:
            for arm, lbl in [("arm_isotropic_per_layer", "per_layer"),
                             ("arm_isotropic_global", "global")]:
                if arm not in stats_by_arm:
                    continue
                b0 = base["prev_only_acc"].mean()
                p0 = pen["prev_only_acc"].mean()
                i0 = stats_by_arm[arm]["prev_only_acc"].mean()
                frac = (i0 - b0) / (p0 - b0) * 100 if p0 != b0 else float("nan")
                print(f"\n  magnitude matching ({lbl}) recovers {frac:.1f}% of the "
                      f"penalty's retention gain "
                      f"(baseline {b0:.4f} -> iso {i0:.4f} -> penalty {p0:.4f})")
                reg.setdefault("recovered_pct", {})[lbl] = float(frac)

        summary["regimes"][regime] = reg

    with open(os.path.join(out, "s1_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n\nwrote {os.path.join(out, 's1_results.json')}")


if __name__ == "__main__":
    main()
