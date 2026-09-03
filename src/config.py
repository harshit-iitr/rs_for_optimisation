"""Run configuration sidecar.

Every run writes config.json BEFORE training starts, so that a run which crashes
mid-way is still fully identifiable. The audit found 47 runs whose only surviving
identity was their directory name.

The sidecar records the complete argparse namespace, every constant that used to
be hardcoded in the training loop, the schema version, the git SHA, the torch
version, and the SHA-256 of any external file the run consumes (e.g. the
isotropic-control target trajectory).
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from src.version import SCHEMA_VERSION


def git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if out.returncode != 0:
            return None
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def file_sha256(path):
    """Hash of an external input file. The audit could not determine which of two
    incompatible iso_shrinkage.json files produced R4_T2c_bp_iso because nothing
    recorded this."""
    if path is None or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Fields that legitimately differ between seeds of the SAME arm. Excluded from
# the comparability hash, but checked separately -- an isotropic run must point at
# its own seed's target, and load_arm verifies that.
SEED_SCOPED = {"seed", "iso_target", "iso_target_sha256"}


def config_hash(cfg):
    """Stable hash over the scientific content of a config, ignoring seed,
    run directory, and timing. Two runs whose config_hash differs are not
    comparable; two runs in the same arm must agree."""
    ignore = SEED_SCOPED | {
        "run_dir", "started_at", "finished_at", "git_sha",
        "torch_version", "python_version", "hostname", "status",
        "wall_clock_sec", "provenance", "n_rows", "n_steps",
        "iso_target_exhausted", "config_hash", "linalg_fallbacks",
        "diverged_at_task", "num_threads",
        "save_final_checkpoint",
    }
    payload = {k: v for k, v in sorted(cfg.items()) if k not in ignore}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def build_config(args, extra=None):
    import torch
    from src.version import CANONICAL

    cfg = dict(vars(args))
    cfg.update(
        schema_version=SCHEMA_VERSION,
        git_sha=git_sha(),
        torch_version=torch.__version__,
        python_version=sys.version.split()[0],
        hostname=os.uname().nodename,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
        status="running",
        provenance="round5",
        canonical_reference=CANONICAL,
    )
    if extra:
        cfg.update(extra)
    cfg["config_hash"] = config_hash(cfg)
    return cfg


def write_config(run_dir, cfg):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def finalize_config(run_dir, status, wall_clock_sec=None, extra=None):
    path = os.path.join(run_dir, "config.json")
    with open(path) as f:
        cfg = json.load(f)
    cfg["status"] = status
    cfg["finished_at"] = datetime.now(timezone.utc).isoformat()
    if wall_clock_sec is not None:
        cfg["wall_clock_sec"] = round(wall_clock_sec, 1)
    if extra:
        cfg.update(extra)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    return cfg
