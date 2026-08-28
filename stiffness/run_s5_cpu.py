import os
import subprocess
import time

commands = []
arms = [
    {'name': 'bp', 'args': '--method bp'},
    {'name': 'cbp', 'args': '--method cbp'},
    {'name': 'ln_l2', 'args': '--method ln_l2 --weight_decay 1e-4'},
    {'name': 'rs', 'args': '--method rs --lambda_rs 1.0'},
]

for arm in arms:
    for seed in [1, 2, 3]:
        run_id = f"S5_2_{arm['name']}_seed{seed}"
        cmd = f"PYTHONPATH=. python3 src/train.py --dataset split_cifar100 --model convnet --n_tasks 10 {arm['args']} --seed {seed} --run_id {run_id}"
        commands.append((run_id, cmd))

running_procs = []
max_concurrent = 6

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = ''
env['OMP_NUM_THREADS'] = '4'

os.makedirs('results', exist_ok=True)

for run_id, cmd in commands:
    parquet_path = os.path.join('results', run_id, 'metrics.parquet')
    if os.path.exists(parquet_path):
        print(f"Skipping {run_id}, already exists.")
        continue
        
    while len(running_procs) >= max_concurrent:
        time.sleep(2)
        running_procs = [p for p in running_procs if p.poll() is None]
        
    print(f"Launching on CPU: {cmd}")
    log_path = os.path.join('results', f"{run_id}.log")
    with open(log_path, 'w') as f:
        p = subprocess.Popen(cmd, shell=True, env=env, stdout=f, stderr=f)
    running_procs.append(p)

while len(running_procs) > 0:
    time.sleep(2)
    running_procs = [p for p in running_procs if p.poll() is None]

print("CPU Sweep S5_2 complete.")
