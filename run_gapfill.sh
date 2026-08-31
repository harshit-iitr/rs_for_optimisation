#!/bin/bash
# Targeted gap-fill: the 10 runs where a single number is currently doing work it
# cannot support. Nothing else in the backlog is touched.
#
#   limit_tangential x3  leg 4b rests on the straight-through variant, not the
#                        true constraint limit
#   lr_0.3 x3            the high-accuracy end of the lr frontier, which leg 2 needs
#   lambda_0.0030 x2     the reported retention optimum, currently n=1
#   lambda_0.0060 x2     its shoulder, also n=1
set -u
cd "$(dirname "$0")"
export PYTHONPATH=.
echo "=== gap-fill started $(date -Is) ==="

python3 -m src.launch --study S3_stiffness_curve --gpus 0,1 --concurrency 7 \
        --threads 16 --per-job-mb 2200 --retries 2 \
        --only limit_tangential,lambda_0.0030,lambda_0.0060 &
S3=$!
sleep 20   # let the first batch claim memory before the second launcher measures
python3 -m src.launch --study S2_lr_frontier --gpus 0,1 --concurrency 3 \
        --threads 16 --per-job-mb 2200 --retries 2 --only lr_0.3 &
S2=$!
wait $S3 $S2

echo ""
echo "=== gap-fill done $(date -Is) ==="
python3 analysis/s2_s3_frontier.py --preliminary 2>&1 | tail -55
echo ""
python3 analysis/s4_equilibrium.py --preliminary 2>&1 | tail -12
python3 -m src.docs > /dev/null 2>&1
