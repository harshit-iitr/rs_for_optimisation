#!/bin/bash
echo "Starting Round 2 Blocking Sweeps"

echo "Running B1: Plasticity Loss..."
python3 src/sweep.py --suite R2_B1

echo "Running B2: Undertrained Grid Check..."
python3 src/sweep.py --suite R2_B2

echo "All Round 2 Blocking Sweeps finished!"
