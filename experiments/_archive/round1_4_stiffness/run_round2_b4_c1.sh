#!/bin/bash
echo "Starting Round 2 Phase 2: Theory Validation (B4 & C1)"

echo "Running B4: 1/lambda equilibrium law..."
python3 src/sweep.py --suite R2_B4

echo "Running C1: The Critical Control (BP Frontier vs RS Curve)..."
python3 src/sweep.py --suite R2_C1

echo "All Phase 2 sweeps finished!"
