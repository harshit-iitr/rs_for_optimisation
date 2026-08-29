"""S2 + S3 -- the two curves, and the paper's legs 1, 2 and 4.

  leg 1  does lowering the step size buy retention, and is it monotone?
  leg 2  does the penalty curve lie ABOVE the learning-rate frontier?
  leg 4  is there an interior optimum, and does the benefit vanish at the
         constraint limit?

Everything is reported on prev_only_acc against test_acc, per seed, with paired
tests. Nothing is averaged across layers.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy import stats
from analysis.common import (EXP, IncompleteArm, config_header, fmt_paired,
                             load_arm, outdir, paired, per_seed)
from src.studies import STUDIES

M = ["test_acc", "prev_only_acc", "avg_seen_acc", "radial_excess",
     "radius_mean", "weight_norm"]


def collect(study, root, label_of):
    out = {}
    base = os.path.join(EXP, root)
    if not os.path.isdir(base):
        return out
    for arm in sorted(os.listdir(base)):
        d = os.path.join(base, arm)
        if not os.path.isdir(d) or arm.startswith("_"):
            continue
        try:
            df = load_arm(d, STUDIES[study]["seeds"])
        except (IncompleteArm, FileNotFoundError, ValueError) as e:
            print(f"  [skipped] {arm}: {str(e)[:130]}")
            continue
        out[label_of(arm)] = per_seed(df, M)
    return out


def curve_table(d, xname):
    rows = []
    for k in sorted(d, key=lambda k: (isinstance(k, str), k)):
        s = d[k]
        rows.append({xname: k, "n": len(s),
                     "test_acc": s.test_acc.mean(), "test_sd": s.test_acc.std(ddof=1),
                     "prev_only": s.prev_only_acc.mean(),
                     "prev_sd": s.prev_only_acc.std(ddof=1),
                     "radial_excess": s.radial_excess.mean(),
                     "weight_norm": s.weight_norm.mean()})
    return pd.DataFrame(rows)


def main():
    out = outdir("S2_S3_frontier")
    res = {}

    print("=" * 78); print("S2 -- LEARNING-RATE FRONTIER"); print("=" * 78)
    lr = collect("S2_lr_frontier", STUDIES["S2_lr_frontier"]["root"],
                 lambda a: float(a.split("lr_")[1]))
    if lr:
        t = curve_table(lr, "lr")
        print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        t.to_csv(os.path.join(out, "s2_lr_frontier.csv"), index=False)
        best = t.loc[t.prev_only.idxmax()]
        print(f"\n  retention is maximised at lr={best.lr} "
              f"(prev_only={best.prev_only:.4f})")
        lo, hi = t.iloc[0], t.iloc[-1]
        print(f"  lowest lr {lo.lr}: prev_only={lo.prev_only:.4f} test={lo.test_acc:.4f}")
        print(f"  highest lr {hi.lr}: prev_only={hi.prev_only:.4f} test={hi.test_acc:.4f}")
        rho, p = stats.spearmanr(t.lr, t.prev_only)
        print(f"  monotone in lr? Spearman rho={rho:+.3f} p={p:.3f} "
              f"({'monotone' if p < 0.05 else 'NOT monotone'})")
        interior = 0 < t.prev_only.idxmax() < len(t) - 1
        print(f"  interior optimum in lr: {interior}")
        res["s2"] = {"table": t.to_dict("records"), "spearman_rho": float(rho),
                     "spearman_p": float(p), "interior_optimum": bool(interior)}

    print("\n" + "=" * 78); print("S3 -- STIFFNESS CURVE"); print("=" * 78)
    def lab(a):
        if a.startswith("lambda_"):
            return float(a.split("lambda_")[1])
        return a
    lam = collect("S3_stiffness_curve", STUDIES["S3_stiffness_curve"]["root"], lab)
    fin = {k: v for k, v in lam.items() if not isinstance(k, str)}
    lim = {k: v for k, v in lam.items() if isinstance(k, str)}
    if fin:
        t = curve_table(fin, "lambda")
        print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        t.to_csv(os.path.join(out, "s3_stiffness_curve.csv"), index=False)
        best = t.loc[t.prev_only.idxmax()]
        interior = 0 < t.prev_only.idxmax() < len(t) - 1
        print(f"\n  LEG 4a: retention optimum at lambda={best['lambda']} "
              f"(prev_only={best.prev_only:.4f}); interior={interior}")
        res["s3"] = {"table": t.to_dict("records"),
                     "lambda_star": float(best["lambda"]),
                     "interior_optimum": bool(interior)}
        if 0.0 in fin:
            print("\n  paired vs lambda=0:")
            for k in sorted(fin):
                if k == 0.0:
                    continue
                print(fmt_paired(paired(fin[0.0], fin[k], "lam=0", f"lam={k}",
                                        "prev_only_acc")))
    for k, s in lim.items():
        print(f"\n  LEG 4b limit arm [{k}]: test={s.test_acc.mean():.4f}"
              f"+-{s.test_acc.std(ddof=1):.4f} "
              f"prev_only={s.prev_only_acc.mean():.4f}+-{s.prev_only_acc.std(ddof=1):.4f} "
              f"radial_excess={s.radial_excess.mean():.4f}")
        if fin and 0.0 in fin:
            print(fmt_paired(paired(fin[0.0], s, "lam=0", k, "prev_only_acc")))
        res.setdefault("limits", {})[k] = {
            "test_acc": float(s.test_acc.mean()),
            "prev_only": float(s.prev_only_acc.mean())}

    if lr and fin:
        print("\n" + "=" * 78)
        print("LEG 2 -- DOES THE PENALTY CURVE LIE ABOVE THE LEARNING-RATE FRONTIER?")
        print("=" * 78)
        lt, st_ = curve_table(lr, "lr"), curve_table(fin, "lambda")
        print(f"  {'lambda':>8s} {'test_acc':>9s} {'prev_only':>10s} | "
              f"{'lr-matched':>10s} {'its prev':>9s} {'gap':>8s}")
        gaps = []
        for _, r in st_.iterrows():
            j = (lt.test_acc - r.test_acc).abs().idxmin()
            m = lt.loc[j]
            gap = r.prev_only - m.prev_only
            gaps.append(gap)
            print(f"  {r['lambda']:8.4f} {r.test_acc:9.4f} {r.prev_only:10.4f} | "
                  f"{m.lr:10.4f} {m.prev_only:9.4f} {gap:+8.4f}")
        res["leg2_max_gap"] = float(np.max(gaps))
        print(f"\n  max vertical gap above the lr frontier: {np.max(gaps):+.4f}")
        print(f"  penalty curve lies above at {int(np.sum(np.array(gaps) > 0))}"
              f"/{len(gaps)} lambda values")

    with open(os.path.join(out, "s2_s3_results.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
