"""Generate the human-navigable layer from the registry and from live run status.

Round 1-4's config audit was a hand-typed Python literal containing a comment
that read "some S1 runs had 5 seeds later? No... I'll put 5." It was wrong about
batch size, learning rate and seed counts (audit section 2.3). Anything derivable
is derived here instead: status comes from each run's own config.json, never from
someone's memory.

The narrative parts -- what the study found, and any caveat -- live between
<!-- FINDING --> markers and are preserved across regeneration.
"""

import json
import os
import re

from src.studies import STUDIES, iter_runs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARK_A, MARK_B = "<!-- FINDING -->", "<!-- /FINDING -->"


def _status(run_dir):
    c = os.path.join(REPO, run_dir, "config.json")
    if not os.path.exists(c):
        return "planned"
    try:
        with open(c) as f:
            st = json.load(f).get("status")
    except Exception:
        return "failed"
    if st == "complete" and os.path.exists(os.path.join(REPO, run_dir, "metrics.parquet")):
        return "complete"
    return "failed" if st == "running" else (st or "failed")


def study_status(name):
    counts = {"complete": 0, "failed": 0, "planned": 0}
    arms = {}
    for _, run_dir, _, _ in iter_runs(name):
        s = _status(run_dir)
        counts[s] = counts.get(s, 0) + 1
        arm = os.path.dirname(run_dir)
        arms.setdefault(arm, []).append((os.path.basename(run_dir), s))
    total = sum(counts.values())
    if counts["complete"] == total:
        overall = "complete"
    elif counts["complete"] == 0 and counts["failed"] == 0:
        overall = "planned"
    elif counts["failed"]:
        overall = "INCOMPLETE (failures present)"
    else:
        overall = "running"
    return overall, counts, arms


def _preserve_finding(path):
    if not os.path.exists(path):
        return ("_Not yet run._", None)
    txt = open(path).read()
    m = re.search(re.escape(MARK_A) + r"(.*?)" + re.escape(MARK_B), txt, re.S)
    return (m.group(1).strip() if m else "_Not yet run._", txt)


def render_study(name):
    st = STUDIES[name]
    overall, counts, arms = study_status(name)
    path = os.path.join(REPO, "experiments", st["root"], "STUDY.md")
    finding, _ = _preserve_finding(path)

    first_args = next(iter_runs(name))[2]
    shared = {k: v for k, v in sorted(first_args.items())
              if k not in ("seed", "method", "lambda_rs", "iso_granularity")}

    L = [f"# {name}", "",
         f"**Question.** {st['question']}", "",
         f"**Supports.** {st['claim']}", "",
         f"**Status.** {overall} — "
         f"{counts['complete']} complete, {counts['failed']} failed, "
         f"{counts['planned']} not started, of {sum(counts.values())} planned runs.", "",
         "## Finding", "", MARK_A, finding, MARK_B, "",
         "## Configuration", "",
         "Shared by every arm unless the arm table says otherwise. Read from the "
         "study registry (`src/studies.py`); each run's actual configuration is in "
         "its own `config.json`.", "",
         "| key | value |", "|---|---|"]
    for k, v in shared.items():
        L.append(f"| `{k}` | `{v}` |")
    L += ["", f"Seeds: `{st['seeds']}`", "",
          "## Arms", "", "| arm | seeds complete | status |", "|---|---|---|"]
    for arm in sorted(arms):
        rows = sorted(arms[arm])
        ok = sum(1 for _, s in rows if s == "complete")
        bad = [r for r, s in rows if s == "failed"]
        note = "complete" if ok == len(rows) else (
            f"FAILED: {', '.join(bad)}" if bad else f"{ok}/{len(rows)} done")
        L.append(f"| `{os.path.relpath(arm, 'experiments/' + st['root'])}` "
                 f"| {ok}/{len(rows)} | {note} |")
    L += ["", "## Phases", ""]
    for i, ph in enumerate(st["phases"]):
        L.append(f"{i}. `{ph['name']}` — {len(ph['arms'])} arm(s)")
    L += ["", "---", "",
          "Reproduce with:", "", "```bash",
          f"python3 -m src.launch --study {name} --tmux", "```", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path, overall, counts


def render_index():
    L = ["# Experiments", "",
         "Round 5. One study per directory; a run's path states what it is, so no "
         "run-id decoding is needed. Every run carries its own `config.json`, "
         "written before training starts.", "",
         "Regenerate this file and every `STUDY.md` with `python3 -m src.docs`. "
         "Status is read from the runs themselves, never typed by hand.", "",
         "| study | question | supports | status | runs |",
         "|---|---|---|---|---|"]
    for name in sorted(STUDIES):
        st = STUDIES[name]
        overall, counts, _ = study_status(name)
        L.append(f"| [`{st['root']}`]({st['root']}/STUDY.md) | {st['question']} "
                 f"| {st['claim']} | {overall} "
                 f"| {counts['complete']}/{sum(counts.values())} |")
    L += ["", "## Layout", "", "```",
          "experiments/",
          "  <benchmark>/<study>/STUDY.md          question, config, arms, status, finding",
          "  <benchmark>/<study>/<arm>/seed_N/     config.json, metrics.parquet,",
          "                                        grad_trace.npz, stdout.log",
          "  _analysis/<study>/                    figures, tables, stats",
          "  _archive/                             Round 1-4. Frozen. Never globbed.",
          "```", "",
          "## Rules", "",
          "1. Every comparison reports a test statistic, p-value, n and sign split; "
          "paired where seeds align.",
          "2. Both axes on every table. `prev_only_acc` is the retention metric; "
          "`avg_seen_acc` is the Round 1-4 quantity, kept for comparability.",
          "3. Per-layer metrics stay per layer. Nothing is ever averaged across layers.",
          "4. Every table is headed with its full config, read from `config.json`.",
          "5. Values identical across arms to four decimals are a bug signal and are "
          "investigated before anything is reported.",
          "6. No arm is reported with fewer seeds than planned. No `nan` standard "
          "deviations.",
          "7. An arm that fails its own acceptance test is not reported at all.",
          "8. Task-loss gradient only in the radial-energy metric.",
          "9. Surprises go to `experiments/ANOMALIES.md`.", ""]
    path = os.path.join(REPO, "experiments", "README.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


if __name__ == "__main__":
    for n in sorted(STUDIES):
        p, o, c = render_study(n)
        print(f"{o:30s} {c['complete']:3d}/{sum(c.values()):3d}  {p}")
    print("index:", render_index())
