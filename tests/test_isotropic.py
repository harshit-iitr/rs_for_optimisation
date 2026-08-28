"""The isotropic control -- the paper's decisive experiment (audit section 3.3.2).

Round 1-4's control scaled gradients and then called clip_grad_norm_ on the next
line, which renormalized the scaling away. It achieved 0.7% of an intended 18.5%
match and nothing in the pipeline noticed. These tests exist so that failure mode
cannot recur silently.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.methods.isotropic import GradTrace, IsotropicControl, acceptance_test
from src.metrics.gradients import group_grad_norms, param_groups, scale_groups_to
from src.models.mlp import MLP


def _model():
    torch.manual_seed(0)
    return MLP(depth=2, width=32, input_dim=16, output_dim=4)


def _populate_grads(model, scale=1.0):
    out = model(torch.randn(8, 16))
    (scale * out.pow(2).sum()).backward()


def test_scaling_hits_the_target_exactly():
    m = _model(); g = param_groups(m); _populate_grads(m)
    targets = [0.1, 0.2, 0.3]
    scale_groups_to(g, targets)
    got, _ = group_grad_norms(g)
    assert np.allclose(got, targets, rtol=1e-5)


def test_scaling_does_not_rotate_the_gradient():
    """An isotropic control transfers magnitude and nothing else."""
    m = _model(); g = param_groups(m); _populate_grads(m)
    before = [p.grad.clone() for _, ps in g for p in ps]
    scale_groups_to(g, [0.1, 0.2, 0.3])
    after = [p.grad.clone() for _, ps in g for p in ps]
    for b, a in zip(before, after):
        cos = torch.nn.functional.cosine_similarity(
            b.flatten(), a.flatten(), dim=0)
        assert cos > 1 - 1e-5


def test_clip_after_scale_destroys_the_match_THE_ROUND_1_4_BUG():
    """Regression test for the exact defect the audit found.

    Scale-then-clip: the clip renormalizes the whole gradient and the intended
    magnitude is gone. Clip-then-scale: the magnitude survives. This is why the
    ordering in src/train.py is load-bearing and asserted at runtime.

    Targets are chosen so the scaled gradient still exceeds the clip threshold,
    which is the regime Round 1-4 was in: its ~0.94 factor was applied to a
    gradient well above the clip, so the clip renormalized it straight back.
    """
    targets = [0.4, 0.4, 0.4]          # global norm 0.69 > clip 0.5

    m = _model(); g = param_groups(m); _populate_grads(m, scale=50.0)
    scale_groups_to(g, targets)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)   # WRONG ORDER
    wrong, _ = group_grad_norms(g)

    m = _model(); g = param_groups(m); _populate_grads(m, scale=50.0)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)   # RIGHT ORDER
    scale_groups_to(g, targets)
    right, _ = group_grad_norms(g)

    assert np.allclose(right, targets, rtol=1e-5), "clip-then-scale must preserve the match"
    assert not np.allclose(wrong, targets, rtol=1e-2), "scale-then-clip must break the match"


def test_control_replays_the_target_trajectory(tmp_path):
    names = ["layer0", "layer1", "head"]
    tr = GradTrace(names)
    rng = np.random.default_rng(0)
    targets = rng.uniform(0.01, 0.2, size=(20, 3))
    for i, row in enumerate(targets):
        tr.record(0, 1.0, row)
    tgt = tmp_path / "grad_trace.npz"; tr.save(tgt)

    iso = IsotropicControl(str(tgt), "per_layer")
    iso.check_compatible(names)

    m = _model(); g = param_groups(m)
    realized_tr = GradTrace(names)
    for step in range(20):
        m.zero_grad(set_to_none=False); _populate_grads(m, scale=1.0 + step)
        realized_tr.record(0, 1.0, iso.apply(g, step))
    out = tmp_path / "realized.npz"; realized_tr.save(out)

    passed, report = acceptance_test(str(out), str(tgt), "per_layer", tol=0.05)
    assert passed, report
    assert report["worst_median_abs_log_ratio"] < 1e-5


def test_acceptance_test_rejects_a_bad_match(tmp_path):
    """The gate must actually reject. Round 1-4 had no gate, so a control that
    achieved 4% of its intended match produced a publishable-looking null."""
    names = ["layer0"]
    a = GradTrace(names); b = GradTrace(names)
    for i in range(50):
        a.record(0, 1.0, [0.10])
        b.record(0, 1.0, [0.10 * 0.85])   # 15% off -- the scale of the R1-4 miss
    pa = tmp_path / "a.npz"; pb = tmp_path / "b.npz"; a.save(pa); b.save(pb)
    passed, report = acceptance_test(str(pa), str(pb), "per_layer", tol=0.05)
    assert not passed
    assert report["worst_median_abs_log_ratio"] > 0.05


def test_incompatible_groups_are_refused(tmp_path):
    tr = GradTrace(["layer0", "layer1", "head"]); tr.record(0, 1.0, [1, 1, 1])
    p = tmp_path / "t.npz"; tr.save(p)
    with pytest.raises(RuntimeError, match="different parameter groups"):
        IsotropicControl(str(p), "per_layer").check_compatible(["layer0", "head"])


def test_groups_cover_every_parameter():
    """Round 1-4 scaled only hidden-layer weights: no biases, no head."""
    m = _model()
    covered = {id(p) for _, ps in param_groups(m) for p in ps}
    assert covered == {id(p) for p in m.parameters()}
