#!/bin/bash
echo "Starting Round 2 Phase 3: Final Benchmark Suites (R1-R6)"

echo "Running R1: Retention-axis lambda sweep..."
python3 src/sweep.py --suite R3_R1

echo "Running R2: Stability Baselines benchmark..."
python3 src/sweep.py --suite R3_R2

echo "Running R4: Width-scaled controls..."
python3 src/sweep.py --suite R3_R4

echo "Running R6: Optimizer retention collapse..."
python3 src/sweep.py --suite R3_R6

echo "All Phase 3 sweeps finished! Ready for final analysis scripts."
