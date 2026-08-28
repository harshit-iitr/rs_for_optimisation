"""Shared analysis loader.

Enforces the reporting rules at load time rather than trusting each script to
remember them. Round 1-4's analysis scripts all guarded with
`if os.path.exists(path)` and silently averaged whatever survived, which is how a
published table came to contain a single-seed mean with a `nan` standard
deviation and how the lambda bracket came to mix 2-seed and 5-seed points without
saying so (audit section 4.4).

Here:
  * an arm with missing seeds RAISES unless explicitly allowed, and the shortfall
    is named
  * every run's config_hash within an arm must agree, so an arm cannot silently
    mix configurations
  * the archive is never reachable -- paths are resolved under experiments/ only,
    and _archive is refused
  * every returned frame carries the config it came from, so tables can be headed
    with a real configuration rather than a reconstructed one
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from src.config import SEED_SCOPED, config_hash


def _first_differing_key(cfgs):
    keys = set().union(*[set(c) for c in cfgs]) - SEED_SCOPED
    for k in sorted(keys):
        vals = {json.dumps(c.get(k), sort_keys=True, default=str) for c in cfgs}
        if len(vals) > 1:
            return f"{k}: {vals}"
    return "none found"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(REPO, "experiments")


class IncompleteArm(Exception):
    pass


def _refuse_archive(path):
    if "_archive" in os.path.normpath(path).split(os.sep):
        raise ValueError(f"analysis must never read the archive: {path}")


def load_run(run_dir):
    _refuse_archive(run_dir)
    d = run_dir if os.path.isabs(run_dir) else os.path.join(REPO, run_dir)
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    if cfg.get("status") != "complete":
        raise IncompleteArm(f"{run_dir}: status={cfg.get('status')}")
    df = pd.read_parquet(os.path.join(d, "metrics.parquet"))
    return cfg, df


def load_arm(arm_dir, expect_seeds, allow_missing=False):
    """Load every seed of one arm. Raises if any planned seed is absent."""
    _refuse_archive(arm_dir)
    frames, cfgs, missing = [], [], []
    for s in expect_seeds:
        rd = os.path.join(arm_dir, f"seed_{s}")
        try:
            cfg, df = load_run(rd)
        except (FileNotFoundError, IncompleteArm) as e:
            missing.append((s, str(e))); continue
        df = df.copy(); df["seed"] = s
        frames.append(df); cfgs.append(cfg)
    if missing and not allow_missing:
        raise IncompleteArm(
            f"{arm_dir}: {len(missing)} of {len(expect_seeds)} seeds missing "
            f"({[m[0] for m in missing]}). Refusing to report a partial arm; "
            f"re-run or pass allow_missing=True and say so in the table.")
    if not frames:
        raise IncompleteArm(f"{arm_dir}: no complete runs")
    # Recomputed, not read from the file, so the check reflects the current
    # definition of "same configuration" rather than whatever was stored.
    hashes = {config_hash(c) for c in cfgs}
    if len(hashes) > 1:
        diff = _first_differing_key(cfgs)
        raise ValueError(f"{arm_dir}: seeds do not share a configuration "
                         f"(hashes {hashes}); first differing key: {diff}")
    # An isotropic run must be matched against its OWN seed's target.
    for c in cfgs:
        tgt = c.get("iso_target")
        if tgt and f"seed_{c['seed']}" not in tgt:
            raise ValueError(
                f"{arm_dir}: seed {c['seed']} is matched against {tgt} -- the "
                f"isotropic control must pair per seed, not across seeds.")
    out = pd.concat(frames, ignore_index=True)
    out.attrs["config"] = cfgs[0]
    out.attrs["n_seeds"] = len(frames)
    out.attrs["missing_seeds"] = [m[0] for m in missing]
    return out


def window(df, layer=None, last_n_tasks=20):
    """Final-window slice. Layer selection is explicit; nothing is ever averaged
    across layers (rule 3)."""
    t_max = df["task"].max()
    out = df[df["task"] > t_max - last_n_tasks]
    if layer is not None:
        out = out[out["layer"] == layer]
    return out


def per_seed(df, cols, layer=0, last_n_tasks=20):
    """One row per seed: the final-window mean of each column at one layer."""
    w = window(df, layer=layer, last_n_tasks=last_n_tasks)
    return w.groupby("seed")[list(cols)].mean().sort_index()


def paired(a, b, label_a, label_b, metric):
    """Paired comparison with test statistic, p, n and sign split (rule 1)."""
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 2:
        return {"metric": metric, "a": label_a, "b": label_b,
                "n": len(common), "note": "too few paired seeds to test"}
    x, y = a.loc[common, metric].values, b.loc[common, metric].values
    t, p = stats.ttest_rel(y, x)
    try:
        _, wp = stats.wilcoxon(y, x)
    except ValueError:
        wp = float("nan")
    d = y - x
    sd = d.std(ddof=1)
    return {
        "metric": metric, "a": label_a, "b": label_b, "n": len(common),
        "mean_a": float(x.mean()), "sd_a": float(x.std(ddof=1)),
        "mean_b": float(y.mean()), "sd_b": float(y.std(ddof=1)),
        "delta": float(d.mean()), "delta_sd": float(sd),
        "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan"),
        "t": float(t), "p": float(p), "wilcoxon_p": float(wp),
        "wins_b_over_a": int((d > 0).sum()), "sign_split": f"{int((d > 0).sum())}/{len(common)}",
    }


def fmt_paired(r):
    if "note" in r:
        return f"  {r['b']} vs {r['a']}: {r['note']} (n={r['n']})"
    return (f"  {r['b']} vs {r['a']} [{r['metric']}]: "
            f"{r['mean_a']:.4f} -> {r['mean_b']:.4f}  "
            f"delta={r['delta']:+.4f}+-{r['delta_sd']:.4f}  "
            f"t={r['t']:+.3f} p={r['p']:.2e} wilcoxon_p={r['wilcoxon_p']:.3f} "
            f"dz={r['cohens_dz']:+.2f} n={r['n']} wins={r['sign_split']}")


def config_header(cfg, keys=None):
    keys = keys or ["dataset", "n_tasks", "epochs", "width", "depth", "act_fn",
                    "lr", "optimizer", "batch_size", "clip_norm", "probe_size",
                    "schema_version", "git_sha"]
    return " | ".join(f"{k}={cfg.get(k)}" for k in keys if k in cfg)


def outdir(study):
    d = os.path.join(EXP, "_analysis", study)
    os.makedirs(d, exist_ok=True)
    return d
