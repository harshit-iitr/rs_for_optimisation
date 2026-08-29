#!/bin/bash
# Round 5 -- run every study back to back, analysing each as it lands.
#
# Order is deliberate: S3 first because it re-measures lambda*, which every other
# study consumes (src/studies.py::LAMBDA_STAR). S2 next because it supplies the
# learning-rate frontier that the paper's leg 2 is measured against. S1 third: it
# is the decisive experiment but its lambda depends on S3.
#
# Nothing here needs supervision. Every study is resumable, config-aware (a
# changed study definition invalidates stale runs), and records its failures.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=.

GPUS=${GPUS:-0,1}
CONC=${CONC:-18}
THREADS=${THREADS:-10}
PERJOB=${PERJOB:-2200}

# CORE first (the paper's four legs), then supporting in value order.
STUDIES="S3_stiffness_curve S2_lr_frontier S1_isotropic_control \
         S9_plasticity_vs_forgetting S6_baselines S8_rotating_mnist \
         S7_optimizers S5_width_scaling"

echo "=========================================================="
echo "ROUND 5 -- full programme"
echo "gpus=$GPUS concurrency=$CONC threads=$THREADS per_job_mb=$PERJOB"
echo "started $(date -Is)"
echo "=========================================================="

for S in $STUDIES; do
  echo ""
  echo "##########################################################"
  echo "## $S   $(date -Is)"
  echo "##########################################################"
  python3 -m src.launch --study "$S" --gpus "$GPUS" --concurrency "$CONC" \
          --threads "$THREADS" --per-job-mb "$PERJOB"

  # analyse immediately, so a result is available the moment it exists
  case "$S" in
    S3_stiffness_curve)
        python3 analysis/s2_s3_frontier.py  2>&1 | tail -60
        python3 analysis/s4_equilibrium.py  2>&1 | tail -25 ;;
    S2_lr_frontier)
        python3 analysis/s2_s3_frontier.py  2>&1 | tail -60 ;;
    S1_isotropic_control)
        python3 analysis/s1_isotropic.py    2>&1 | tail -80 ;;
    *)  python3 analysis/sweeps.py "$S"     2>&1 | tail -40 ;;
  esac
  python3 -m src.docs > /dev/null 2>&1
  echo "## $S done $(date -Is)"
done

echo ""
echo "=========================================================="
echo "ALL STUDIES DONE $(date -Is)"
echo "=========================================================="
python3 -m src.docs
echo ""
echo "--- failures across all studies ---"
python3 - <<'PY'
import json, glob
bad = 0
for f in sorted(glob.glob("experiments/**/_launch_report.json", recursive=True)):
    r = json.load(open(f))
    if r.get("failed"):
        bad += len(r["failed"])
        print(f"{r['study']}: {len(r['failed'])} failed")
        for x in r["failed"][:10]:
            print("   ", x)
print("no failures" if bad == 0 else f"TOTAL FAILED: {bad}")
PY
