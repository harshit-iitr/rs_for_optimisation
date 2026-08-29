"""The isotropic (magnitude-matched) control.

This is the paper's decisive experiment. Round 1-4's version was invalid: it
multiplied hidden-layer weight gradients by a factor derived from END-OF-TASK
WEIGHT NORMS, averaged over five seeds, and then called clip_grad_norm_ on the
next line, which renormalized the scaling away. Intended weight-norm reduction
18.5%, achieved 0.7%; the null result it produced was read as proof of
directionality (audit section 3.3.2).

Every one of those faults is corrected here:

  target quantity   realized PER-STEP gradient magnitude, logged during the
                    penalty run itself -- not weight norms, not a probe statistic
  ordering          scaling is applied AFTER clipping, so it survives; the
                    ordering is asserted at runtime, not assumed
  coverage          all parameters -- every layer's weight AND bias, and the
                    output head. Round 1-4 scaled only hidden-layer weights
  pairing           per seed against that seed's own penalty run, not against a
                    population mean over seeds
  granularity       per step, per group. Round 1-4 used one scalar per (task, layer)
  verification      the realized trajectory is logged and must pass an acceptance
                    test BEFORE any retention number is looked at

Rather than applying a ratio (which drifts as the trajectories diverge), the
control sets each group's gradient norm directly to the target run's recorded
norm. The match is then exact by construction, and the acceptance test verifies
the plumbing rather than hoping for it.

Direction is never touched. Only magnitude is transferred from the penalty arm.
"""

import numpy as np

from src.metrics.gradients import scale_global_to, scale_groups_to


class GradTrace:
    """Per-step record of realized gradient magnitudes for one run."""

    def __init__(self, group_names):
        self.group_names = list(group_names)
        self.post = []       # realized per-group norms, after clip and any scaling
        self.pre_global = [] # global norm before clipping -- shows whether the clip binds
        self.wnorm = []      # per-group WEIGHT norms after the step
        self.task_of_step = []

    def record(self, task, pre_global, post_group_norms, weight_norms=None):
        self.task_of_step.append(task)
        self.pre_global.append(pre_global)
        self.post.append(list(post_group_norms))
        if weight_norms is not None:
            self.wnorm.append(list(weight_norms))

    def save(self, path):
        d = dict(
            group_names=np.array(self.group_names, dtype=object),
            post=np.asarray(self.post, dtype=np.float32),
            pre_global=np.asarray(self.pre_global, dtype=np.float32),
            task_of_step=np.asarray(self.task_of_step, dtype=np.int32),
        )
        if self.wnorm:
            d["wnorm"] = np.asarray(self.wnorm, dtype=np.float32)
        np.savez_compressed(path, **d)


class IsotropicControl:
    """Replays a target run's realized gradient magnitudes onto this run."""

    def __init__(self, target_npz, granularity="per_layer"):
        if granularity not in ("per_layer", "global"):
            raise ValueError(f"unknown granularity {granularity!r}")
        z = np.load(target_npz, allow_pickle=True)
        self.granularity = granularity
        self.group_names = [str(s) for s in z["group_names"]]
        self.post = z["post"]                       # (n_steps, n_groups)
        self.global_norm = np.linalg.norm(self.post, axis=1)
        self.n_steps = self.post.shape[0]
        self.exhausted = False

    def check_compatible(self, group_names):
        if list(group_names) != self.group_names:
            raise RuntimeError(
                "isotropic target was recorded with different parameter groups:\n"
                f"  target: {self.group_names}\n  this run: {list(group_names)}"
            )

    def target_global(self, step_idx):
        if step_idx >= self.n_steps:
            return None
        return float(self.global_norm[step_idx])

    def apply(self, groups, step_idx):
        """Set this step's gradient magnitude to the target run's. Must be called
        AFTER clipping. Returns the realized per-group norms."""
        if step_idx >= self.n_steps:
            self.exhausted = True
            from src.metrics.gradients import group_grad_norms
            return group_grad_norms(groups)[0]
        from src.metrics.gradients import group_grad_norms
        if self.granularity == "per_layer":
            return scale_groups_to(groups, list(self.post[step_idx]))
        scale_global_to(groups, float(self.global_norm[step_idx]))
        return group_grad_norms(groups)[0]


class WeightNormControl:
    """Match the penalty run's per-layer WEIGHT-norm trajectory, isotropically.

    The gradient-magnitude control answers "is the penalty an anisotropic
    step-size schedule?". It does not answer the other obvious objection: the
    penalty leaves the network at a smaller weight norm (45.3 vs 50.6) and a
    smaller activation radius (43.1 vs 49.8), and maybe *that* is what buys
    retention. This arm tests it directly: after every step, each parameter group
    is rescaled so its weight norm equals the penalty run's at that step.

    Direction in weight space is untouched -- only the radius of each group is
    set. If retention is recovered here, the mechanism is norm, not direction.
    """

    def __init__(self, target_npz):
        z = np.load(target_npz, allow_pickle=True)
        if "wnorm" not in z:
            raise RuntimeError(
                f"{target_npz} has no weight-norm trace; re-run the target arm "
                f"with --log_grad_trace under the current trainer.")
        self.group_names = [str(x) for x in z["group_names"]]
        self.wnorm = z["wnorm"]
        self.n_steps = self.wnorm.shape[0]
        self.exhausted = False

    def check_compatible(self, group_names):
        if list(group_names) != self.group_names:
            raise RuntimeError("weight-norm target has different parameter groups")

    def target(self, step_idx):
        if step_idx >= self.n_steps:
            self.exhausted = True
            return None
        return self.wnorm[step_idx]

    def apply(self, groups, step_idx):
        import torch
        tgt = self.target(step_idx)
        if tgt is None:
            return None
        with torch.no_grad():
            for (_, ps), t in zip(groups, tgt):
                sq = sum(float(p.detach().pow(2).sum()) for p in ps)
                cur = sq ** 0.5
                if cur > 1e-12:
                    f = float(t) / cur
                    for p in ps:
                        p.detach().mul_(f)
        return tgt


def acceptance_test(realized_npz, target_npz, granularity, tol=0.05):
    """Did the control actually match the magnitude it claims to?

    Pre-registered criterion, applied to whatever the arm CLAIMS to match:

      granularity="per_layer"  the median absolute log-ratio must be below `tol`
                               (default 0.05) globally AND within every group
      granularity="global"     it must be below `tol` for the global norm; the
                               per-group residual is reported but is not a
                               pass/fail criterion, because a global-norm control
                               does not claim to match per-group magnitudes

    The per-group residual is reported for both, since for the global arm it is
    exactly the portion of the magnitude confound that arm leaves behind.

    Returns (passed, report_dict). If this fails, the arm is not reported at all.
    """
    r = np.load(realized_npz, allow_pickle=True)
    t = np.load(target_npz, allow_pickle=True)
    rn, tn = r["post"], t["post"]
    n = min(len(rn), len(tn))
    rn, tn = rn[:n], tn[:n]

    def med_abs_log_ratio(a, b):
        m = (a > 1e-12) & (b > 1e-12)
        if m.sum() == 0:
            return float("nan"), 0
        return float(np.median(np.abs(np.log(a[m] / b[m])))), int(m.sum())

    report = {
        "granularity": granularity,
        "tolerance": tol,
        "n_steps_compared": int(n),
        "n_steps_realized": int(len(r["post"])),
        "n_steps_target": int(len(t["post"])),
        "step_count_match": bool(len(r["post"]) == len(t["post"])),
        "per_group": {},
    }
    g_stat, g_n = med_abs_log_ratio(
        np.linalg.norm(rn, axis=1), np.linalg.norm(tn, axis=1)
    )
    report["global_median_abs_log_ratio"] = g_stat
    report["global_n_compared"] = g_n

    names = [str(s) for s in t["group_names"]]
    worst = g_stat
    for i, name in enumerate(names):
        s, cnt = med_abs_log_ratio(rn[:, i], tn[:, i])
        report["per_group"][name] = {"median_abs_log_ratio": s, "n": cnt}
        if not np.isnan(s):
            worst = max(worst, s)

    report["worst_median_abs_log_ratio"] = worst
    report["worst_per_group_median_abs_log_ratio"] = (
        max((v["median_abs_log_ratio"] for v in report["per_group"].values()
             if not np.isnan(v["median_abs_log_ratio"])), default=float("nan")))

    criterion = worst if granularity == "per_layer" else g_stat
    report["criterion_value"] = criterion
    report["criterion"] = ("max over global and all groups" if granularity == "per_layer"
                           else "global norm only")
    report["passed"] = bool(
        report["step_count_match"] and not np.isnan(criterion) and criterion < tol
    )
    return report["passed"], report
