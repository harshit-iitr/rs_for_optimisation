import os
import subprocess
import time

def main():
    commands = []
    
    # B1 commands
    for lr in [0.01, 0.1]:
        for seed in [1, 2, 3]:
            cmd = f"PYTHONPATH=. python3 src/train.py --method bp --lr {lr} --seed {seed} --n_tasks 300 --run_id B1_lr_{lr}_seed{seed}"
            commands.append(cmd)
            
    # B2 commands
    for ep in [1, 3, 10]:
        for seed in [1, 2, 3]:
            cmd = f"PYTHONPATH=. python3 src/train.py --method bp --lr 0.01 --seed {seed} --n_tasks 50 --epochs {ep} --log_within_task --run_id B2_ep_{ep}_seed{seed}"
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
        
    print("B1 & B2 complete.")

if __name__ == '__main__':
    main()
