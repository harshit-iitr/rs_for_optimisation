"""Retention metrics.

The audit found Round 1-4's `prev_tasks_acc` averaged over all tasks seen so far
INCLUDING the current one -- it is the standard average-seen-accuracy (ACC), not
previous-task retention. The definition was uniform across every arm and schema
generation, so archived comparisons are internally valid, but the name overstated
it and the bias favours arms with higher current-task accuracy, which is the very
thing "at matched current-task accuracy" is trying to control for.

Round 5 logs both, under accurate names:
  * avg_seen_acc  -- the Round 1-4 quantity, unchanged, for comparability.
  * prev_only_acc -- previous tasks only, NaN at task 0. The paper's retention metric.
"""

import math

import torch


@torch.no_grad()
def evaluate_tasks(model, task_tests, probe_size, forward_kwargs=None, task_incremental=False, dataset=None):
    """Accuracy on each task seen so far, in task order.

    task_tests: list of (x_test, y_test), index 0 == task 0, last == current task.
    Evaluates the first `probe_size` test samples of each task, identically for
    every arm.
    """
    forward_kwargs = forward_kwargs or {}
    was_training = model.training
    model.eval()
    accs = []
    for task_idx, (px, py) in enumerate(task_tests):
        logits = model(px[:probe_size], **forward_kwargs)
        
        if task_incremental and dataset == "split_mnist":
            # Mask out all classes except 2*task_idx and 2*task_idx + 1
            mask = torch.full_like(logits, float('-inf'))
            mask[:, 2 * task_idx] = 0
            mask[:, 2 * task_idx + 1] = 0
            logits = logits + mask
            
        accs.append((logits.argmax(dim=1) == py[:probe_size]).float().mean().item())
    if was_training:
        model.train()
    return accs


def summarize(accs):
    """accs: per-task accuracies including the current task at the end."""
    if not accs:
        return {"avg_seen_acc": float("nan"), "prev_only_acc": float("nan"),
                "task_0_acc": float("nan"), "n_tasks_seen": 0}
    prev = accs[:-1]
    return {
        "avg_seen_acc": sum(accs) / len(accs),
        "prev_only_acc": (sum(prev) / len(prev)) if prev else float("nan"),
        "task_0_acc": accs[0],
        "n_tasks_seen": len(accs),
    }
