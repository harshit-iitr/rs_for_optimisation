"""Generic per-study report for the supporting sweeps (S5-S9).

One table per study on both axes, every arm paired against its own control, with
test statistic, p, n and sign split. Arms with missing seeds are refused, not
averaged (analysis/common.py).
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import (EXP, IncompleteArm, config_header, fmt_paired,
                             load_arm, outdir, paired, per_seed)
from src.studies import STUDIES

M = ["test_acc", "prev_only_acc", "avg_seen_acc", "task_0_acc",
     "weight_norm", "radius_mean", "radial_excess"]


def arm_dirs(root):
    base = os.path.join(EXP, root)
    found = []
    for dirpath, dirnames, _ in os.walk(base):
        if any(x.startswith("seed_") for x in dirnames):
            found.append(dirpath)
    return sorted(found)


def report(study):
    st = STUDIES[study]
    out = outdir(study)
    print("=" * 78); print(study); print("=" * 78)
    print(f"question: {st['question']}")
    arms, cfg0 = {}, None
    for d in arm_dirs(st["root"]):
        name = os.path.relpath(d, os.path.join(EXP, st["root"]))
        try:
            df = load_arm(d, st["seeds"])
        except (IncompleteArm, FileNotFoundError, ValueError) as e:
            print(f"  [NOT REPORTABLE] {name}: {str(e)[:140]}"); continue
        arms[name] = per_seed(df, M)
        cfg0 = cfg0 or df.attrs["config"]
    if not arms:
        print("  no complete arms"); return None
    print(f"\nconfig: {config_header(cfg0)}\n")
    rows = [{"arm": k, "n": len(s),
             "test_acc": s.test_acc.mean(), "test_sd": s.test_acc.std(ddof=1),
             "prev_only": s.prev_only_acc.mean(),
             "prev_sd": s.prev_only_acc.std(ddof=1),
             "weight_norm": s.weight_norm.mean(),
             "radius": s.radius_mean.mean()} for k, s in arms.items()]
    t = pd.DataFrame(rows).sort_values("prev_only", ascending=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    t.to_csv(os.path.join(out, f"{study}_table.csv"), index=False)

    # pair every arm against the study's natural control
    ctl = next((k for k in arms if k in ("baseline", "arm_baseline")), None)
    tests = []
    if ctl:
        print(f"\npaired vs '{ctl}' [prev_only_acc]:")
        for k in arms:
            if k == ctl:
                continue
            r = paired(arms[ctl], arms[k], ctl, k, "prev_only_acc")
            print(fmt_paired(r)); tests.append(r)
    else:
        # width/optimizer studies: pair penalty against baseline within each cell
        for k in sorted(arms):
            if k.endswith("/penalty"):
                b = k.replace("/penalty", "/baseline")
                if b in arms:
                    r = paired(arms[b], arms[k], b, k, "prev_only_acc")
                    print(fmt_paired(r)); tests.append(r)
    with open(os.path.join(out, f"{study}_results.json"), "w") as f:
        json.dump({"table": t.to_dict("records"), "tests": tests}, f,
                  indent=2, default=float)
    return t


if __name__ == "__main__":
    if "--preliminary" in sys.argv:
        sys.argv.remove("--preliminary")
        from analysis.common import set_preliminary
        set_preliminary(True)
        from analysis.common import banner; banner("supporting sweeps")

    todo = sys.argv[1:] or ["S5_width_scaling", "S6_baselines",
                            "S7_optimizers", "S8_rotating_mnist",
                            "S9_plasticity_vs_forgetting"]
    for s in todo:
        try:
            report(s); print()
        except Exception as e:
            print(f"{s}: analysis failed: {e}\n")
