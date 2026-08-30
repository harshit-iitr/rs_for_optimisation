"""S4 -- the equilibrium-radius law. No new training; reads S3's runs.

Theory predicts the steady-state radial excess scales as 1/lambda, i.e. a log-log
slope of -1. Round 1-4 found the slope was much flatter and depth-dependent, and
recorded that as a failed prediction (archive ANOMALIES.md).

The refinement tested here: the deficit is accounted for by the radial component
of the task gradient itself varying with lambda. If u* ~ g_rad / lambda and
g_rad ~ lambda^a, then slope(u*) = a - 1. So the measured exponent of g_rad,
minus one, should reproduce the measured slope of u*, LAYER BY LAYER.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy import stats
from analysis.common import EXP, IncompleteArm, load_arm, outdir, per_seed
from src.studies import STUDIES

STUDY = "S3_stiffness_curve"


def main():
    out = outdir("S4_equilibrium")
    base = os.path.join(EXP, STUDIES[STUDY]["root"])
    rows = []
    for arm in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if not arm.startswith("lambda_"):
            continue
        lam = float(arm.split("lambda_")[1])
        if lam <= 0:
            continue
        try:
            df = load_arm(os.path.join(base, arm), STUDIES[STUDY]["seeds"])
        except (IncompleteArm, FileNotFoundError, ValueError) as e:
            print(f"  [skipped] {arm}: {str(e)[:110]}"); continue
        for layer in sorted(df.layer.unique()):
            s = per_seed(df, ["radial_excess", "g_rad_norm", "g_norm_task"],
                         layer=int(layer))
            rows.append({"lam": lam, "layer": int(layer),
                         "radial_excess": s.radial_excess.mean(),
                         "g_rad_norm": s.g_rad_norm.mean(),
                         "n": len(s)})
    if not rows:
        print("no S3 data yet"); return
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "s4_raw.csv"), index=False)

    print("=" * 78)
    print("S4 -- EQUILIBRIUM LAW  (log-log fits, per layer; prediction: slope -1)")
    print("=" * 78)
    print(f"{'layer':>5s} {'n_lam':>6s} {'slope(u*)':>11s} {'ci95':>8s} "
          f"{'slope(g_rad)=a':>15s} {'a-1':>8s} {'residual':>9s}")
    res = []
    for layer in sorted(d.layer.unique()):
        g = d[(d.layer == layer) & (d.radial_excess > 0)].sort_values("lam")
        if len(g) < 3:
            print(f"{layer:5d}  too few positive-excess points ({len(g)})"); continue
        x = np.log10(g.lam.values)
        fu = stats.linregress(x, np.log10(g.radial_excess.values))
        fg = stats.linregress(x, np.log10(g.g_rad_norm.values))
        ci = fu.stderr * stats.t.ppf(0.975, len(x) - 2)
        resid = fu.slope - (fg.slope - 1.0)
        print(f"{layer:5d} {len(g):6d} {fu.slope:11.4f} {ci:8.4f} "
              f"{fg.slope:15.4f} {fg.slope - 1:8.4f} {resid:9.4f}")
        res.append({"layer": int(layer), "n_lambda": len(g),
                    "slope_u": float(fu.slope), "ci95": float(ci),
                    "slope_grad": float(fg.slope),
                    "predicted_slope": float(fg.slope - 1.0),
                    "residual": float(resid),
                    "naive_prediction_minus1_within_ci":
                        bool(abs(fu.slope + 1.0) <= ci),
                    "refined_prediction_within_ci": bool(abs(resid) <= ci)})
    print("\n  'slope(u*)' is the measured log-log slope of radial excess vs lambda.")
    print("  Naive theory predicts -1. The refined prediction is a-1, where a is")
    print("  the measured exponent of the radial task-gradient component.")
    for r in res:
        print(f"    layer {r['layer']}: naive -1 within CI = "
              f"{r['naive_prediction_minus1_within_ci']}, "
              f"refined a-1 within CI = {r['refined_prediction_within_ci']}")
    with open(os.path.join(out, "s4_fits.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    if "--preliminary" in sys.argv:
        sys.argv.remove("--preliminary")
        from analysis.common import set_preliminary
        set_preliminary(True)
        from analysis.common import banner; banner("S4 equilibrium law")

    main()
