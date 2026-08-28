"""Retention metric definitions (audit section 3.3.4)."""
import math
from src.metrics.retention import summarize


def test_avg_seen_includes_current_task():
    """Preserves the Round 1-4 quantity bit-for-bit so archived runs stay
    comparable -- but under its accurate name."""
    assert summarize([0.9, 0.5, 0.4])["avg_seen_acc"] == (0.9 + 0.5 + 0.4) / 3


def test_prev_only_excludes_current_task():
    assert summarize([0.9, 0.5, 0.4])["prev_only_acc"] == (0.9 + 0.5) / 2


def test_prev_only_is_nan_at_task_zero():
    assert math.isnan(summarize([0.9])["prev_only_acc"])
    assert summarize([0.9])["avg_seen_acc"] == 0.9


def test_the_two_differ_when_current_task_is_easier():
    """The bias that motivated splitting them: avg_seen_acc is inflated by the
    current task, which favours arms with higher current-task accuracy -- exactly
    what 'at matched current-task accuracy' is trying to control for."""
    s = summarize([0.2] * 9 + [0.96])
    assert s["avg_seen_acc"] > s["prev_only_acc"]
