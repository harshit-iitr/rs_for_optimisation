"""Paper figures, generated from the Round-5 runs on disk.

Colour follows the validated reference palette from the dataviz skill, assigned
in fixed slot order by series identity and never cycled. Text is in ink tokens,
never in a series colour. No dual axes: the ratchet is two panels because weight
norm and activation radius are different measures on different scales.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.common import EXP, load_arm, per_seed, set_preliminary
set_preliminary(True)

SC = os.path.join(EXP, "permuted_mnist/stiffness_curve")
LR = os.path.join(EXP, "permuted_mnist/lr_frontier")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "latex", "figs")

# validated categorical palette, light surface, fixed slot order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "lines.linewidth": 1.3, "lines.markersize": 3.2,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

def style(ax):
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.9)   # solid, recessive
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def arm(root, name, seeds=(1, 2, 3)):
    try:
        return load_arm(os.path.join(root, name), list(seeds))
    except Exception:
        return None

def traj(name, metric, layer=0):
    d = arm(SC, name)
    if d is None:
        return None, None
    g = d[d.layer == layer].groupby("task")[metric]
    return g.mean().index.values, g.mean().values

# ------------------------------------------------------------------ ratchet
def fig_ratchet():
    arms = [("lambda_0.0000", r"$\lambda=0$ (baseline)", C[0]),
            ("lambda_0.0030", r"$\lambda=3{\times}10^{-3}$", C[1]),
            ("lambda_10.0000", r"$\lambda=10$", C[2]),
            ("limit_tangential", "hard projection", C[3])]
    # vertical nudges (points) per arm per panel, to keep labels from overprinting
    dy = {"weight_norm":     {"lambda_0.0000": 0, "lambda_0.0030": 0,
                              "lambda_10.0000": 0, "limit_tangential": 0},
          "radius_mean":     {"lambda_0.0000": 0, "lambda_0.0030": 0,
                              "lambda_10.0000": -5, "limit_tangential": 5}}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.05))
    for ax, metric, lab in [(axes[0], "weight_norm", r"$\|W\|_F$  (layer 0)"),
                            (axes[1], "radius_mean", r"activation radius $\|h\|_2$")]:
        for name, label, col in arms:
            t, y = traj(name, metric)
            if t is None:
                continue
            ax.plot(t, y, color=col, label=label)
            ax.annotate(label, xy=(t[-1], y[-1]), xytext=(4, dy[metric][name]),
                        textcoords="offset points", va="center",
                        fontsize=5.8, color=INK2)
        ax.set_xlabel("task"); ax.set_ylabel(lab)
        ax.set_xlim(0, 232)
        style(ax)
    axes[1].axhline(np.sqrt(1000), color=INK2, linewidth=0.5, zorder=0)
    axes[1].annotate(r"$\sqrt{d}$", xy=(18, np.sqrt(1000)), xytext=(0, -9),
                     textcoords="offset points", ha="center",
                     fontsize=5.8, color=INK2)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_ratchet.pdf")); plt.close(fig)
    print("  fig_ratchet.pdf")

# -------------------------------------------------------------- equilibrium
def fig_equilibrium():
    lams = [1e-4, 3e-3, 6e-3, 1e-2, 1e-1, 1.0, 10.0]
    names = {1e-4: "lambda_0.0001", 3e-3: "lambda_0.0030", 6e-3: "lambda_0.0060",
             1e-2: "lambda_0.0100", 1e-1: "lambda_0.1000", 1.0: "lambda_1.0000",
             10.0: "lambda_10.0000"}
    fig, ax = plt.subplots(figsize=(3.1, 2.05))
    for layer, col in [(0, C[0]), (1, C[1])]:
        xs, ys = [], []
        for lam in lams:
            d = arm(SC, names[lam])
            if d is None:
                continue
            u = per_seed(d, ["radial_excess"], layer=layer).radial_excess.mean()
            if u > 0:
                xs.append(lam); ys.append(u)
        if len(xs) < 3:
            continue
        ax.plot(xs, ys, "o-", color=col, label=f"layer {layer}")
        f = stats.linregress(np.log10(xs), np.log10(ys))
        anchor = {0: (xs[0], ys[0], (6, -11)), 1: (xs[-1], ys[-1], (-4, 8))}[layer]
        ax.annotate(f"fitted slope {f.slope:+.2f}", xy=(anchor[0], anchor[1]),
                    xytext=anchor[2], textcoords="offset points",
                    ha="left" if layer == 0 else "right",
                    fontsize=5.8, color=col)
    xr = np.array([3e-4, 3e-2])
    ax.plot(xr, 6.0 * (xr / 3e-4) ** (-1.0), color=INK2, linewidth=0.7, zorder=0)
    ax.annotate("slope $-1$ (theory)", xy=(xr[1], 6.0 * (xr[1] / 3e-4) ** -1.0),
                xytext=(3, -3), textcoords="offset points",
                fontsize=5.8, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"penalty strength $\lambda$")
    ax.set_ylabel(r"radial excess $u^\star$")
    ax.legend(frameon=False, loc="lower left")
    style(ax); fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_equilibrium.pdf")); plt.close(fig)
    print("  fig_equilibrium.pdf")

# ----------------------------------------------------------------- frontier
def fig_frontier():
    lam_arms = [("lambda_0.0000", 0.0), ("lambda_0.0001", 1e-4),
                ("lambda_0.0030", 3e-3), ("lambda_0.0060", 6e-3),
                ("lambda_0.0100", 1e-2), ("lambda_0.1000", 1e-1),
                ("lambda_1.0000", 1.0), ("lambda_10.0000", 10.0)]
    fig, ax = plt.subplots(figsize=(3.1, 2.05))
    keep = {}
    for root, arms_, col, lab, mk in [
            (LR, [(f"lr_{v}", v) for v in [0.003, 0.01, 0.025, 0.05, 0.1, 0.3]],
             C[1], "step size $\\eta$", "s"),
            (SC, lam_arms, C[0], "penalty $\\lambda$", "o")]:
        xs, ys, vs = [], [], []
        for name, v in arms_:
            d = arm(root, name, seeds=(1, 2, 3))
            if d is None:
                continue
            s = per_seed(d, ["test_acc", "prev_only_acc"], layer=0)
            xs.append(s.test_acc.mean()); ys.append(s.prev_only_acc.mean()); vs.append(v)
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], mk + "-", color=col,
                label=lab, markersize=2.8)
        keep[lab] = (np.array(xs), np.array(ys), np.array(vs, dtype=float))
        # selective direct labels only: the optimum and the far endpoint
        for i in (int(np.argmax(ys)), int(np.argmin(xs))):
            ax.annotate(f"{vs[i]:g}", xy=(xs[i], ys[i]), xytext=(4, 3),
                        textcoords="offset points", fontsize=5.6, color=col)
    # the claim, drawn: vertical gap at the accuracy the step size can reach
    if len(keep) == 2:
        px, py, _ = keep["penalty $\\lambda$"]
        lx, ly, _ = keep["step size $\\eta$"]
        j = int(np.argmax(py)); i = int(np.argmin(np.abs(lx - px[j])))
        ax.annotate("", xy=(px[j], py[j]), xytext=(px[j], ly[i]),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7))
        ax.annotate(f"{py[j]-ly[i]:+.3f}", xy=(px[j], (py[j]+ly[i])/2),
                    xytext=(-5, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=6.0, color=INK)
    ax.set_xlabel("current-task accuracy")
    ax.set_ylabel("retention")
    ax.legend(frameon=False, loc="upper left")
    style(ax); fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_frontier.pdf")); plt.close(fig)
    print("  fig_frontier.pdf")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_ratchet(); fig_equilibrium(); fig_frontier()
