# S10_selectivity

**Question.** Are hidden units task-shared or task-specific, and does the penalty move the network between those regimes?

**Supports.** Supporting (mechanism, descriptive). Supplies the selectivity figure.

**Status.** complete — 12 complete, 0 failed, 0 not started, of 12 planned runs.

## Finding

<!-- FINDING -->
**The penalty makes layer-1 units markedly more task-shared. The hard-projection limits do not.**

Per-unit normalised response over 150 tasks, per layer, n=3 seeds. `selectivity_index` is
0 = responds equally to every task, → 1 = responds to essentially one task.
`participation_ratio` (PR) is the effective number of tasks a unit serves.

| layer | arm | selectivity_index | PR (of 150) | vs baseline | p |
|---|---|---|---|---|---|
| 0 | baseline | 0.7041 | 101.4 | — | — |
| 0 | penalty | 0.7003 | 102.9 | −0.0038 | 0.041 |
| 0 | limit_tangential | 0.7106 | 100.4 | +0.0066 | 0.011 |
| 0 | limit_ste | 0.7061 | 101.2 | +0.0020 | 0.137 |
| **1** | **baseline** | **0.9502** | **18.2** | — | — |
| **1** | **penalty** | **0.8313** | **68.9** | **−0.1189** | **0.0004** |
| 1 | limit_tangential | 0.9361 | 24.4 | −0.0142 | 0.012 |
| 1 | limit_ste | 0.9460 | 19.8 | −0.0042 | 0.077 |
| 2 | baseline | 0.7062 | 82.2 | — | — |
| 2 | penalty | 0.6293 | 112.7 | −0.0770 | 0.027 |
| 2 | limit_tangential | 0.6561 | 106.7 | −0.0501 | 0.124 |
| 2 | limit_ste | 0.6440 | 90.1 | −0.0623 | 0.056 |

All sign splits 3/3 for the penalty arm. `dead_frac` is 0 in every arm, so none of this is a
dead-unit artifact — the units are all alive, the question was only what they respond to.

**Layer 1 is where the conditions separate, and the separation is large.** In the baseline a
layer-1 unit serves ~18 of 150 tasks; under the penalty it serves ~69, nearly a 4× increase in
participation, at index 0.950 → 0.831 (p=0.0004, the largest effect in the battery). Layer 0 is
unchanged by anything (~101 tasks in every arm) — the input layer stays task-shared regardless.
Layer 2 shifts in the same direction as layer 1 but weakly and with high variance.

**The hard-projection limits move layer 1 barely at all** (−0.014 tangential, −0.004 ste). This
is the same dissociation the stiffness curve shows for retention: the soft penalty produces the
effect and the enforced constraint does not. Together with T6, the representational signature
and the retention benefit appear and disappear under the same conditions.

Whether this *explains* retention is not settled here. The correlation across conditions is
consistent with "the penalty protects old tasks by spreading them across shared units rather
than overwriting task-specific ones", but this study is descriptive: 4 conditions, no
intervention that manipulates sharing independently of the penalty. The causal version is T11.

**Figure.** `latex/figs/selectivity_layer{0,1,2}.pdf`, units × tasks, one panel per condition,
rows sorted by preferred task. **Caption must note that the bright diagonal is an artifact of
per-unit max normalisation** — every unit has exactly one task at 1.0 by construction, and the
diagonal appears identically in all panels. The signal is the background density, which is
visibly higher in the penalty panel.

12/12 runs complete, 0 failed, 12 checkpoints saved. `save_final_checkpoint` is excluded from
`config_hash` (src/config.py), verified invariant on 12 existing configs, so these arms remain
comparable with the S3 arms at the same configuration.
<!-- /FINDING -->

## Configuration

Shared by every arm unless the arm table says otherwise. Read from the study registry (`src/studies.py`); each run's actual configuration is in its own `config.json`.

| key | value |
|---|---|
| `act_fn` | `relu` |
| `batch_size` | `256` |
| `clip_norm` | `0.5` |
| `dataset` | `permuted_mnist` |
| `depth` | `3` |
| `epochs` | `10` |
| `lr` | `0.1` |
| `n_tasks` | `150` |
| `optimizer` | `sgd` |
| `probe_size` | `2000` |
| `save_final_checkpoint` | `True` |
| `spectral_every` | `5` |
| `width` | `1000` |

Seeds: `[1, 2, 3]`

## Arms

| arm | seeds complete | status |
|---|---|---|
| `baseline` | 3/3 | complete |
| `limit_ste` | 3/3 | complete |
| `limit_tangential` | 3/3 | complete |
| `penalty` | 3/3 | complete |

## Phases

0. `sweep` — 4 arm(s)

---

Reproduce with:

```bash
python3 -m src.launch --study S10_selectivity --tmux
```
