"""Resumable, memory-aware study launcher.

Round 1-4 lost 47 runs to CUDA OOM from fixed over-concurrency on a shared GPU,
and every analysis script silently skipped the gaps (audit section 4.4). Three
things prevent that here:

  * concurrency is gated on MEASURED free GPU memory, not a hardcoded job count
  * a run is resumed by checking its own config.json status, so re-launching a
    partially-completed study costs nothing
  * failures are recorded in _launch_report.json; analysis refuses to report an
    arm with missing seeds rather than averaging what survived

Phases run in order and a phase does not start until the previous one is complete,
which is what makes the isotropic control's two-pass dependency safe.

Usage
  python3 -m src.launch --study S1_isotropic_control --dry-run
  python3 -m src.launch --study S1_isotropic_control --tmux
  python3 -m src.launch --study S1_isotropic_control --measure
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

from src.studies import STUDIES, iter_runs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOL_FLAGS = {"track_drift", "log_grad_trace"}


def gpu_free_mb(allowed=None):
    """Free MB per visible GPU, restricted to `allowed` device ids if given."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        free = [int(x) for x in out.stdout.split()]
    except Exception:
        return {}
    ids = allowed if allowed is not None else list(range(len(free)))
    return {i: free[i] for i in ids if i < len(free)}


def run_status(run_dir, expect_args=None):
    """'complete', 'stale', 'diverged', 'failed', or None if never started.

    'stale' means the run finished but under a DIFFERENT configuration than the
    study now defines. Resuming past a stale run is how a study silently ends up
    mixing configurations -- the failure mode that produced eight incompatible
    schema generations in Rounds 1-4 -- so it is treated as not-done.
    """
    cpath = os.path.join(run_dir, "config.json")
    if not os.path.exists(cpath):
        return None
    try:
        with open(cpath) as f:
            cfg = json.load(f)
    except Exception:
        return "failed"
    st = cfg.get("status")
    if st == "diverged":
        return "diverged"
    if st != "complete" or not os.path.exists(os.path.join(run_dir, "metrics.parquet")):
        return "failed" if st == "running" else (st or "failed")
    if expect_args is not None:
        want = {k: v for k, v in expect_args.items()}
        for k, v in want.items():
            got = cfg.get(k)
            if isinstance(v, float) or isinstance(got, float):
                try:
                    if abs(float(got) - float(v)) > 1e-12:
                        return "stale"
                    continue
                except (TypeError, ValueError):
                    return "stale"
            if got != v:
                return "stale"
    return "complete"


def build_cmd(run_dir, args, iso_target):
    parts = [sys.executable, "-u", "-m", "src.train", "--run_dir", run_dir]
    for k, v in sorted(args.items()):
        if k in BOOL_FLAGS:
            if v:
                parts.append(f"--{k}")
        else:
            parts += [f"--{k}", str(v)]
    if iso_target:
        parts += ["--iso_target", os.path.join(iso_target, "grad_trace.npz")]
    return parts


def launch_study(study_name, concurrency, reserve_mb, per_job_mb, dry_run,
                 force, threads=8, gpus=None, retries=2, only=None):
    st = STUDIES[study_name]
    plan = list(iter_runs(study_name))
    n_phases = len(st["phases"])
    report = {"study": study_name, "started_at": time.time(),
              "complete": [], "skipped": [], "failed": []}
    rpath = os.path.join(REPO, "experiments", st["root"], "_launch_report.json")
    os.makedirs(os.path.dirname(rpath), exist_ok=True)

    for phase_i in range(n_phases):
        todo = []
        for pi, run_dir, args, dep in plan:
            if pi != phase_i:
                continue
            # --only restricts the run to named arms, so a targeted gap-fill does
            # not drag the whole study's backlog along with it.
            if only and not any(o in run_dir for o in only):
                continue
            abs_dir = os.path.join(REPO, run_dir)
            status = run_status(abs_dir, args)
            if status == "stale":
                print(f"  STALE (config changed since it ran), will re-run: {run_dir}")
            if status == "complete" and not force:
                report["skipped"].append(run_dir)
                continue
            if dep and run_status(os.path.join(REPO, dep)) != "complete":
                print(f"  BLOCKED (dependency incomplete): {run_dir}")
                report["failed"].append({"run": run_dir, "reason": "dependency incomplete"})
                continue
            todo.append((run_dir, args, dep))

        pname = st["phases"][phase_i]["name"]
        print(f"\n=== phase {phase_i} ({pname}): {len(todo)} to run, "
              f"{len(report['skipped'])} already complete ===")
        if dry_run:
            for run_dir, args, dep in todo[:4]:
                print("   ", " ".join(shlex.quote(c) for c in build_cmd(run_dir, args, dep)))
            if len(todo) > 4:
                print(f"    ... and {len(todo) - 4} more")
            continue

        running = []
        for run_dir, args, dep in todo:
            abs_dir = os.path.join(REPO, run_dir)
            os.makedirs(abs_dir, exist_ok=True)

            while True:
                running = [(p, d, g) for p, d, g in running if p.poll() is None]
                free = gpu_free_mb(gpus)
                gpu = None
                if free:
                    # Balance by JOBS IN FLIGHT, not by free memory. A freshly
                    # launched job takes seconds to allocate, so picking the
                    # emptiest GPU repeatedly piles everything onto one of them
                    # before any of it registers.
                    onit = {g: sum(1 for _, _, gg in running if gg == g) for g in free}
                    eligible = [g for g in free
                                if free[g] - reserve_mb >= per_job_mb + onit[g] * per_job_mb]
                    if not eligible:
                        eligible = [g for g in free
                                    if free[g] - reserve_mb >= per_job_mb]
                    if eligible:
                        gpu = min(eligible, key=lambda g: (onit[g], -free[g]))
                if len(running) < concurrency and (gpu is not None or not free):
                    break
                time.sleep(5)

            # Set before the child imports torch, so BLAS picks them up.
            env = dict(os.environ, PYTHONPATH=REPO,
                       OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                       OPENBLAS_NUM_THREADS=str(threads))
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = build_cmd(run_dir, args, dep) + ["--num_threads", str(threads)]
            print(f"  [gpu {gpu}] {run_dir}")
            with open(os.path.join(abs_dir, "stdout.log"), "w") as f:
                p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
            running.append((p, run_dir, gpu))
            time.sleep(6)   # stagger allocator spikes; also lets memory register

        while running:
            running = [(p, d, g) for p, d, g in running if p.poll() is None]
            time.sleep(5)

        # Retry pass. On a shared GPU a run can be OOM-killed by another user's
        # allocation through no fault of its own; one attempt is not enough.
        for attempt in range(1, retries + 1):
            again = [(rd, a, d) for rd, a, d in todo
                     if run_status(os.path.join(REPO, rd), a) != "complete"]
            if not again:
                break
            print(f"  --- retry {attempt}/{retries}: {len(again)} run(s) ---")
            running = []
            for run_dir, args, dep in again:
                abs_dir = os.path.join(REPO, run_dir)
                while True:
                    running = [(p, d, g) for p, d, g in running if p.poll() is None]
                    free = gpu_free_mb(gpus)
                    gpu = None
                    if free:
                        onit = {g: sum(1 for _, _, gg in running if gg == g) for g in free}
                        el = [g for g in free if free[g] - reserve_mb >= per_job_mb]
                        if el:
                            gpu = min(el, key=lambda g: (onit[g], -free[g]))
                    if len(running) < max(1, concurrency // 2) and (gpu is not None or not free):
                        break
                    time.sleep(5)
                env = dict(os.environ, PYTHONPATH=REPO, OMP_NUM_THREADS=str(threads),
                           MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
                if gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                cmd = build_cmd(run_dir, args, dep) + ["--num_threads", str(threads)]
                print(f"  [retry gpu {gpu}] {run_dir}")
                with open(os.path.join(abs_dir, "stdout.log"), "w") as f:
                    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f,
                                         stderr=subprocess.STDOUT)
                running.append((p, run_dir, gpu))
                time.sleep(6)
            while running:
                running = [(p, d, g) for p, d, g in running if p.poll() is None]
                time.sleep(5)

        for _, run_dir, exp_args, _ in [x for x in plan if x[0] == phase_i]:
            s = run_status(os.path.join(REPO, run_dir), exp_args)
            if s == "complete":
                if run_dir not in report["skipped"]:
                    report["complete"].append(run_dir)
            else:
                report["failed"].append({"run": run_dir, "reason": f"status={s}"})

        with open(rpath, "w") as f:
            json.dump(report, f, indent=2)

    if not dry_run:
        report["finished_at"] = time.time()
        with open(rpath, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n=== {study_name}: {len(report['complete'])} complete, "
              f"{len(report['skipped'])} skipped, {len(report['failed'])} FAILED ===")
        for fr in report["failed"]:
            print("   FAILED:", fr)
        print("report:", rpath)
    return report


def measure(study_name, tasks=3):
    """Time and memory-profile one canonical run of this study, then report the
    projected total. Concurrency is set from this, not from optimism."""
    _, run_dir, args, _ = next(iter_runs(study_name))
    args = dict(args); args["n_tasks"] = tasks
    args.pop("log_grad_trace", None)
    probe = os.path.join(REPO, "experiments", "_measure", study_name)
    os.makedirs(probe, exist_ok=True)
    cmd = build_cmd(probe, args, None)
    print("measuring:", " ".join(shlex.quote(c) for c in cmd))

    before = list(gpu_free_mb().values())
    t0 = time.time()
    env = dict(os.environ, PYTHONPATH=REPO)
    p = subprocess.Popen(cmd, cwd=REPO, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak_used = 0
    while p.poll() is None:
        now = list(gpu_free_mb().values())
        if before and now:
            peak_used = max(peak_used, max(b - n for b, n in zip(before, now)))
        time.sleep(2)
    out = p.stdout.read()
    dt = time.time() - t0
    if p.returncode != 0:
        print(out[-3000:])
        raise SystemExit(f"measurement run failed (rc={p.returncode})")

    full = next(iter_runs(study_name))[2]["n_tasks"]
    per_run = dt / tasks * full
    n = sum(1 for _ in iter_runs(study_name))
    print(f"\n  {tasks} tasks in {dt:.1f}s  ->  ~{per_run/3600:.2f} h per full "
          f"{full}-task run")
    print(f"  peak GPU memory for one job: ~{peak_used} MB")
    print(f"  {n} runs in this study -> {n*per_run/3600:.1f} GPU-hours")
    tot = sum(sum(1 for _ in iter_runs(s)) for s in STUDIES)
    print(f"  {tot} runs across all studies -> ~{tot*per_run/3600:.0f} GPU-hours "
          f"(studies differ in cost; this extrapolates from {study_name})")
    if peak_used > 0:
        free = list(gpu_free_mb().values())
        cap = sum(max(0, f - 2000) // max(peak_used, 1) for f in free)
        print(f"  free now {free} MB -> safe concurrency ~{cap} "
              f"(2 GB/GPU reserve)")
    return per_run, peak_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=sorted(STUDIES) + ["all"])
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--reserve-mb", type=int, default=2000,
                    help="GPU memory left free at all times")
    ap.add_argument("--per-job-mb", type=int, default=4000,
                    help="expected peak per job; set from --measure")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run completed runs")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated substrings; run only matching arms")
    ap.add_argument("--retries", type=int, default=2,
                    help="re-attempt runs that failed; a shared GPU can OOM-kill "
                         "a run through no fault of its own")
    ap.add_argument("--gpus", type=str, default=None,
                    help="comma-separated device ids to use, e.g. '0'. "
                         "Default: all visible GPUs.")
    ap.add_argument("--threads", type=int, default=8,
                    help="CPU intra-op threads per job; concurrency*threads "
                         "should not exceed nproc")
    ap.add_argument("--tmux", action="store_true",
                    help="re-exec detached under tmux so disconnects cannot kill it")
    a = ap.parse_args()

    if a.tmux:
        session = f"r5_{a.study}"
        inner = [sys.executable, "-u", "-m", "src.launch"] + [
            x for x in sys.argv[1:] if x != "--tmux"]
        cmd = " ".join(shlex.quote(c) for c in inner)
        subprocess.run(["tmux", "kill-session", "-t", session],
                       capture_output=True)
        subprocess.run(["tmux", "new-session", "-d", "-s", session,
                        f"cd {shlex.quote(REPO)} && PYTHONPATH={shlex.quote(REPO)} "
                        f"{cmd} 2>&1 | tee experiments/_launch_{a.study}.log"],
                       check=True)
        print(f"launched in tmux session '{session}'")
        print(f"  attach: tmux attach -t {session}")
        print(f"  log:    experiments/_launch_{a.study}.log")
        return

    if a.measure:
        measure(a.study)
        return

    for s in (sorted(STUDIES) if a.study == "all" else [a.study]):
        gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else None
        only = [x.strip() for x in a.only.split(",")] if a.only else None
        launch_study(s, a.concurrency, a.reserve_mb, a.per_job_mb, a.dry_run,
                     a.force, a.threads, gpus, a.retries, only)


if __name__ == "__main__":
    main()
