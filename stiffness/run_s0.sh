#!/bin/bash
PYTHONPATH=. python3 src/train.py --lambda_rs 0.0 --seed 1 --n_tasks 150 --run_id S0.2_seed1 &
PYTHONPATH=. python3 src/train.py --lambda_rs 0.0 --seed 2 --n_tasks 150 --run_id S0.2_seed2 &
PYTHONPATH=. python3 src/train.py --lambda_rs 0.0 --seed 3 --n_tasks 150 --run_id S0.2_seed3 &
PYTHONPATH=. python3 src/train.py --lambda_rs 0.05 --seed 1 --n_tasks 20 --run_id S0.3_seed1 &
wait
echo "All done!"
