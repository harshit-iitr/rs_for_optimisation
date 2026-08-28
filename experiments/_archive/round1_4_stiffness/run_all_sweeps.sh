#!/bin/bash
set -e
echo "Starting S4.3 Sweep"
python3 src/sweep.py --suite S4_3
echo "Starting S5.1 Sweep"
python3 src/sweep.py --suite S5_1
echo "Starting S5.2 Sweep"
python3 src/sweep.py --suite S5_2
echo "Starting S5.3 Sweep"
python3 src/sweep.py --suite S5_3
echo "Starting S6.1 Sweep"
python3 src/sweep.py --suite S6_1
echo "Starting S6.2 Sweep"
python3 src/sweep.py --suite S6_2

echo "Running S4.2 Timescale"
PYTHONPATH=. python3 analysis/s4_2_timescale.py

echo "Running S3 Diagnostic Race Analysis"
PYTHONPATH=. python3 analysis/s3_diagnostic_race.py

echo "Running S6.3 Stability Tax Analysis"
PYTHONPATH=. python3 analysis/s6_3_stability_tax.py

echo "All sweeps and analyses completed successfully!"
