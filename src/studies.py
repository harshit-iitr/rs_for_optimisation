"""Declarative study registry.

Round 1-4 encoded its sweeps as hardcoded branches inside sweep.py, so a run's
configuration existed only in whichever version of that file was on disk that day
(audit section 4.1). Here a study is data: the launcher materialises it into the
experiment tree and every run carries its own config.json.

A run's directory path states what it is. No run-id decoding.
"""

from src.version import CANONICAL

# The stiffness-curve optimum used by every arm that needs "the penalty at its
# best". Taken from the archived R4_T1 sweep, which had 5 clean seeds at the
# current schema and peaked here. S3 re-measures it; if S3 moves it, this changes
# and the affected studies are re-run.
LAMBDA_STAR = 0.003


def _canon(**over):
    cfg = dict(
        dataset=CANONICAL["dataset"], n_tasks=CANONICAL["n_tasks"],
        epochs=CANONICAL["epochs"], width=CANONICAL["width"],
        depth=CANONICAL["depth"], act_fn=CANONICAL["act_fn"], lr=CANONICAL["lr"],
        optimizer=CANONICAL["optimizer"], batch_size=CANONICAL["batch_size"],
        clip_norm=CANONICAL["clip_norm"], probe_size=CANONICAL["probe_size"],
        spectral_every=5,
    )
    cfg.update(over)
    return cfg


def _lam_dir(lam):
    return f"lambda_{lam:.4f}" if lam else "lambda_0.0000"


STUDIES = {}


# Seed policy. Five seeds only where a paired test between two arms is itself
# the result -- that is S1, the decisive control. The curve-shaped studies get
# their statistical power from the number of arms along the curve rather than
# from seeds within an arm, so three is sufficient and buys back the compute.
SEEDS_DECISIVE = [1, 2, 3, 4, 5]
SEEDS_CURVE = [1, 2, 3]

def study(name, root, question, claim, seeds, phases):
    STUDIES[name] = dict(name=name, root=root, question=question, claim=claim,
                         seeds=seeds, phases=phases)


# ---------------------------------------------------------------- S1
# The decisive experiment. Two phases per regime: the baseline and penalty arms
# record their realized per-step gradient magnitudes AND weight norms; the
# control arms then replay the penalty arm's magnitudes onto the baseline.
#
# THREE controls, because they answer three different objections:
#   isotropic_per_layer  matches realized gradient magnitude per parameter group.
#                        Rules out "the penalty is an anisotropic step-size schedule".
#   isotropic_global     matches only the total gradient norm. Under clipping this
#                        is already equal, so it doubles as a null control: it
#                        should reproduce the baseline exactly.
#   iso_wnorm            matches the penalty's per-group WEIGHT-norm trajectory.
#                        Rules out "the penalty just parks the net at a smaller
#                        norm". The gradient controls do NOT reproduce the
#                        penalty's weight-norm drop (45.3 vs 50.6) so without this
#                        arm the objection stands open.
#
# TWO clipping regimes, both stable:
#   clipped  clip_norm=0.5, canonical. Binds on 89% of steps and pins BOTH arms to
#            a global gradient norm of exactly 0.500, so the global magnitude
#            confound is 0.00% and only the per-layer confound (up to 10.7%) is
#            real here.
#   loose    clip_norm=10, which almost never binds and leaves the global
#            magnitude confound exposed, but still catches the rare gradient
#            spikes that make the fully-unclipped run diverge.
#
# A fully unclipped regime was measured and DISCARDED: at lr=0.1 it diverges
# (activation radius 45 -> 253 by task ~10, then 130 tasks of chance accuracy).
# Probes: no-clip/lr0.1 diverges; clip10/lr0.1 0.971; clip2/lr0.1 0.968;
# no-clip/lr0.03 0.961. Clipping is load-bearing for stability at lr=0.1.
_S1_TARGETS = [
    ("arm_baseline", dict(method="bp", lambda_rs=0.0)),
    (f"arm_penalty_lam{LAMBDA_STAR}", dict(method="rs", lambda_rs=LAMBDA_STAR)),
]
_S1_CONTROLS = [
    ("arm_isotropic_per_layer", dict(method="isotropic", iso_granularity="per_layer")),
    ("arm_isotropic_global", dict(method="isotropic", iso_granularity="global")),
    ("arm_iso_wnorm", dict(method="iso_wnorm")),
]


def _s1_phases():
    phases = []
    for regime, clip in [("clipped", 0.5), ("loose", 10.0)]:
        phases.append({"name": f"{regime}_A_targets", "arms": [
            {"dir": f"{regime}/{d}",
             "args": _canon(clip_norm=clip, track_drift=True, log_grad_trace=True, **a)}
            for d, a in _S1_TARGETS]})
        phases.append({"name": f"{regime}_B_controls", "arms": [
            {"dir": f"{regime}/{d}",
             "args": _canon(lambda_rs=0.0, clip_norm=clip, track_drift=True,
                            log_grad_trace=True, **a),
             "iso_target_from": f"{regime}/arm_penalty_lam{LAMBDA_STAR}"}
            for d, a in _S1_CONTROLS]})
    return phases


study(
    "S1_isotropic_control",
    root="permuted_mnist/isotropic_control",
    question="Is the penalty's retention benefit reproduced by an equal-magnitude "
             "or equal-weight-norm intervention that carries no directional "
             "information?",
    claim="Paper claim 3 (mechanism: direction vs magnitude). Decisive.",
    seeds=SEEDS_DECISIVE,
    phases=_s1_phases(),
)

# ---------------------------------------------------------------- S2
study(
    "S2_lr_frontier",
    root="permuted_mnist/lr_frontier",
    question="Does lowering the step size buy retention, and is the relationship "
             "monotone?",
    claim="Paper claim 1. Becomes the reference frontier for claim 2.",
    seeds=SEEDS_CURVE,
    phases=[{"name": "sweep", "arms": [
        {"dir": f"lr_{lr}", "args": _canon(method="bp", lambda_rs=0.0, lr=lr,
                                           track_drift=True)}
        for lr in [0.003, 0.01, 0.015, 0.025, 0.05, 0.1, 0.3]
    ]}],
)

# ---------------------------------------------------------------- S3
study(
    "S3_stiffness_curve",
    root="permuted_mnist/stiffness_curve",
    question="Where is the retention optimum in penalty strength, and does the "
             "benefit vanish once the constraint is enforced?",
    claim="Paper claims 2 and 4; supplies the data for the equilibrium-law fit (S4).",
    seeds=SEEDS_CURVE,
    phases=[{"name": "sweep", "arms": [
        {"dir": _lam_dir(lam),
         "args": _canon(method="rs" if lam else "bp", lambda_rs=lam, track_drift=True)}
        for lam in [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.006, 0.01, 0.03,
                    0.1, 1.0, 10.0]
    ] + [
        # The two limit arms are different algorithms and are never aliased.
        {"dir": "limit_tangential",
         "args": _canon(method="bp", lambda_rs=0.0, projection="tangential",
                        track_drift=True)},
        {"dir": "limit_ste",
         "args": _canon(method="bp", lambda_rs=0.0, projection="ste",
                        track_drift=True)},
    ]}],
)

# ---------------------------------------------------------------- S5
study(
    "S5_width_scaling",
    root="permuted_mnist/width_scaling",
    question="Does the penalty's advantage grow with width?",
    claim="Supporting. Round 1-4's version lacked per-width baselines at the "
          "canonical budget.",
    seeds=SEEDS_CURVE,
    phases=[{"name": "sweep", "arms": [
        {"dir": f"width_{w}/{'penalty' if lam else 'baseline'}",
         "args": _canon(method="rs" if lam else "bp", lambda_rs=lam, width=w)}
        for w in [256, 512, 1000, 2048] for lam in [0.0, LAMBDA_STAR]
    ]}],
)

# ---------------------------------------------------------------- S6
# Dropped vs Round 1-4: continual backprop (never ran -- audit 5.1), the
# effective-rank penalty and its combination (they destroy the network -- audit
# 5.5a). MAS is retained but flagged: its retention reflects a network that never
# trained (test accuracy 0.41).
study(
    "S6_baselines",
    root="permuted_mnist/baselines",
    question="Where does the penalty sit on the plasticity-stability frontier "
             "relative to parameter-regularization baselines, and does it compose?",
    claim="Supporting (Pareto position and additivity).",
    seeds=SEEDS_CURVE,
    phases=[{"name": "sweep", "arms": [
        {"dir": "baseline", "args": _canon(method="bp")},
        {"dir": "weight_decay_1e-3", "args": _canon(method="l2", weight_decay=1e-3)},
        {"dir": "weight_decay_1e-4", "args": _canon(method="l2", weight_decay=1e-4)},
        {"dir": "layernorm_wd_1e-4", "args": _canon(method="ln_l2", weight_decay=1e-4)},
        {"dir": "l2_init", "args": _canon(method="l2_init")},
        {"dir": "shrink_perturb", "args": _canon(method="sp")},
        {"dir": "ewc_100", "args": _canon(method="ewc", ewc_lambda=100)},
        {"dir": "ewc_1000", "args": _canon(method="ewc", ewc_lambda=1000)},
        {"dir": "ewc_10000", "args": _canon(method="ewc", ewc_lambda=10000)},
        {"dir": "si", "args": _canon(method="si")},
        {"dir": "mas", "args": _canon(method="mas")},
        {"dir": "penalty", "args": _canon(method="rs", lambda_rs=LAMBDA_STAR)},
        {"dir": "penalty_plus_ewc_10000",
         "args": _canon(method="rs_ewc", lambda_rs=LAMBDA_STAR, ewc_lambda=10000)},
    ]}],
)

# ---------------------------------------------------------------- S7
# Round 1-4 compared arms across two code generations at a 10x learning-rate
# difference, and its two adaptive arms were bitwise identical because
# AdamW(wd=0) == Adam(wd=0) (audit 5.3). Only one adaptive optimizer is run here,
# and the per-optimizer learning rate is stated rather than inferred.
study(
    "S7_optimizers",
    root="permuted_mnist/optimizers",
    question="Does the effect survive momentum, and does it survive adaptive "
             "preconditioning?",
    claim="Supporting. Single code generation; lr stated per optimizer.",
    seeds=SEEDS_CURVE,
    phases=[{"name": "sweep", "arms": [
        {"dir": f"{opt}/{'penalty' if lam else 'baseline'}",
         "args": _canon(method="rs" if lam else "bp", lambda_rs=lam,
                        optimizer=opt, lr=lr)}
        for opt, lr in [("sgd", 0.1), ("sgd_momentum", 0.01), ("adam", 0.001)]
        for lam in [0.0, LAMBDA_STAR]
    ]}],
)

# ---------------------------------------------------------------- S8
# Round 1-4 tested only lambda in {1, 10} here -- both far past the optimum.
study(
    "S8_rotating_mnist",
    root="rotating_mnist/stiffness_curve",
    question="Does the stiffness curve reproduce on a second benchmark?",
    claim="Supporting (generality).",
    seeds=SEEDS_CURVE,
    phases=[{"name": "sweep", "arms": [
        {"dir": _lam_dir(lam),
         "args": _canon(method="rs" if lam else "bp", lambda_rs=lam,
                        dataset="rotating_mnist", n_tasks=100, epochs=1,
                        width=256, depth=2, lr=0.01)}
        for lam in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
    ]}],
)

# ---------------------------------------------------------------- S9
study(
    "S9_plasticity_vs_forgetting",
    root="permuted_mnist/plasticity_vs_forgetting",
    question="Which failure mode does this benchmark exhibit at the canonical "
             "config -- loss of plasticity or forgetting?",
    claim="Setup. Establishes what the paper is measuring.",
    seeds=SEEDS_CURVE,
    phases=[{"name": "sweep", "arms": [
        {"dir": "baseline_300tasks", "args": _canon(method="bp", n_tasks=300)},
        {"dir": "penalty_300tasks",
         "args": _canon(method="rs", lambda_rs=LAMBDA_STAR, n_tasks=300)},
    ]}],
)


def iter_runs(study_name, exp_root="experiments"):
    """Yield (phase_index, run_dir, args_dict, iso_target_dir_or_None) in
    dependency order."""
    st = STUDIES[study_name]
    for pi, phase in enumerate(st["phases"]):
        for arm in phase["arms"]:
            for seed in st["seeds"]:
                run_dir = f"{exp_root}/{st['root']}/{arm['dir']}/seed_{seed}"
                args = dict(arm["args"]); args["seed"] = seed
                dep = arm.get("iso_target_from")
                dep_dir = (f"{exp_root}/{st['root']}/{dep}/seed_{seed}"
                           if dep else None)
                yield pi, run_dir, args, dep_dir


def total_runs(study_name):
    return sum(1 for _ in iter_runs(study_name))


# ---------------------------------------------------------------- S10
# Checkpointed re-runs of the canonical conditions, for analysis/selectivity.py.
# Same configuration as the corresponding S3 arms; the only difference is that
# these write final_model.pt. save_final_checkpoint is I/O only and is excluded
# from config_hash (src/config.py), so it does not fork the comparability class.
study(
    "S10_selectivity",
    root="permuted_mnist/selectivity",
    question="Are hidden units task-shared or task-specific, and does the "
             "penalty move the network between those regimes?",
    claim="Supporting (mechanism, descriptive). Supplies the selectivity figure.",
    seeds=SEEDS_CURVE,
    phases=[{"name": "sweep", "arms": [
        {"dir": "baseline",
         "args": _canon(method="bp", lambda_rs=0.0, save_final_checkpoint=True)},
        {"dir": "penalty",
         "args": _canon(method="rs", lambda_rs=LAMBDA_STAR, save_final_checkpoint=True)},
        {"dir": "limit_tangential",
         "args": _canon(method="bp", lambda_rs=0.0, projection="tangential",
                        save_final_checkpoint=True)},
        {"dir": "limit_ste",
         "args": _canon(method="bp", lambda_rs=0.0, projection="ste",
                        save_final_checkpoint=True)},
    ]}],
)
