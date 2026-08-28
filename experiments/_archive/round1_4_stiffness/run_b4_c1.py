import os
import subprocess
import time

def main():
    commands = []
    
    # B4 commands (50 tasks, 10 epochs)
    lambdas_b4 = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]
    for lam in lambdas_b4:
        for seed in [1, 2, 3]:
            cmd = f"PYTHONPATH=. python3 src/train.py --method rs --lambda_rs {lam} --lr 0.01 --seed {seed} --n_tasks 50 --epochs 10 --run_id B4_lam_{lam}_seed{seed}"
            commands.append(cmd)
            
    # C1 Curve A commands (BP, 150 tasks, 10 epochs)
    lrs_c1 = [0.003, 0.01, 0.03, 0.1, 0.3]
    for lr in lrs_c1:
        for seed in [1, 2, 3, 4, 5]:
            cmd = f"PYTHONPATH=. python3 src/train.py --method bp --lr {lr} --seed {seed} --n_tasks 150 --epochs 10 --run_id C1_A_lr_{lr}_seed{seed}"
            commands.append(cmd)

    # C1 Curve B commands (RS, lr=0.1, 150 tasks, 10 epochs)
    lambdas_c1 = [0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    for lam in lambdas_c1:
        for seed in [1, 2, 3, 4, 5]:
            method = "bp" if lam == 0 else "rs"
            cmd = f"PYTHONPATH=. python3 src/train.py --method {method} --lambda_rs {lam} --lr 0.1 --seed {seed} --n_tasks 150 --epochs 10 --run_id C1_B_lam_{lam}_seed{seed}"
            commands.append(cmd)

    n_gpus = 2
    jobs_per_gpu = 4
    max_concurrent = n_gpus * jobs_per_gpu
    
    running_procs = []
    
    for i, cmd in enumerate(commands):
        gpu_id = i % n_gpus
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        while len(running_procs) >= max_concurrent:
            time.sleep(1)
            running_procs = [(p, cmd_str) for p, cmd_str in running_procs if p.poll() is None]
            
        print(f"Launching on GPU {gpu_id}: {cmd}")
        log_file = cmd.split("--run_id ")[1].split(" ")[0] + ".log"
        log_path = os.path.join('results', log_file)
        os.makedirs('results', exist_ok=True)
        
        with open(log_path, 'w') as f:
            p = subprocess.Popen(cmd, shell=True, env=env, stdout=f, stderr=f)
        running_procs.append((p, cmd))
        
        time.sleep(2)
        
    while len(running_procs) > 0:
        time.sleep(1)
        running_procs = [(p, cmd_str) for p, cmd_str in running_procs if p.poll() is None]
        
    print("B4 & C1 complete.")

if __name__ == '__main__':
    main()
