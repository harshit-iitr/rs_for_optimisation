import argparse
import subprocess
import os
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=str, required=True, help="Suite name (e.g. S1)")
    args = parser.parse_args()

    commands = []
    
    if args.suite == 'S1':
        lambdas = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 'inf']
        seeds = [1, 2, 3]
        for lam in lambdas:
            for seed in seeds:
                if lam == 'inf':
                    cmd = f"PYTHONPATH=. python3 src/train.py --hard_projection --seed {seed} --n_tasks 150 --run_id S1_lam_inf_seed{seed}"
                else:
                    cmd = f"PYTHONPATH=. python3 src/train.py --lambda_rs {lam} --seed {seed} --n_tasks 150 --run_id S1_lam_{lam}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'S2':
        seeds = [1, 2, 3, 4, 5]
        arms = [
            {'name': 'bp', 'args': '--method bp'},
            {'name': 'l2_1e-3', 'args': '--method l2 --weight_decay 1e-3'},
            {'name': 'l2_1e-4', 'args': '--method l2 --weight_decay 1e-4'},
            {'name': 'ln', 'args': '--method ln'},
            {'name': 'ln_l2', 'args': '--method ln_l2 --weight_decay 1e-4'},
            {'name': 'sp', 'args': '--method sp'},
            {'name': 'l2_init', 'args': '--method l2_init'},
            {'name': 'cbp', 'args': '--method cbp'},
            {'name': 'redo', 'args': '--method redo'},
            {'name': 'er', 'args': '--method er'},
            {'name': 'l2_er', 'args': '--method l2_er --weight_decay 1e-4'},
            {'name': 'rs', 'args': '--method rs --lambda_rs 1.0'}
        ]
        for arm in arms:
            for seed in seeds:
                cmd = f"PYTHONPATH=. python3 src/train.py {arm['args']} --seed {seed} --n_tasks 150 --run_id S2_{arm['name']}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'S4_3':
        for w in [256, 512, 1000, 2048]:
            for seed in [1, 2, 3]:
                cmd = f"PYTHONPATH=. python3 src/train.py --method rs --lambda_rs 1.0 --width {w} --seed {seed} --n_tasks 150 --run_id S4_3_width_{w}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'S5_1':
        for lam in [0.0, 1.0, 10.0]:
            for seed in [1, 2, 3]:
                cmd = f"PYTHONPATH=. python3 src/train.py --dataset rotating_mnist --model mlp --depth 2 --width 256 --n_tasks 100 --lambda_rs {lam} --method rs --seed {seed} --run_id S5_1_rot_lam_{lam}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'S5_2':
        arms = [
            {'name': 'bp', 'args': '--method bp'},
            {'name': 'ln_l2', 'args': '--method ln_l2 --weight_decay 1e-4'},
            {'name': 'rs', 'args': '--method rs --lambda_rs 1.0'}
        ]
        for arm in arms:
            for seed in [1, 2, 3]:
                cmd = f"PYTHONPATH=. python3 src/train.py --dataset split_cifar100 --model convnet --n_tasks 10 {arm['args']} --seed {seed} --run_id S5_2_{arm['name']}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'S5_3':
        for act in ['relu', 'leaky_relu']:
            for lam in [0.0, 1.0]:
                for seed in [1, 2, 3]:
                    cmd = f"PYTHONPATH=. python3 src/train.py --act_fn {act} --method rs --lambda_rs {lam} --seed {seed} --n_tasks 150 --run_id S5_3_{act}_lam_{lam}_seed{seed}"
                    commands.append(cmd)

    elif args.suite == 'S6_1':
        for lr in [1e-3, 1e-2, 1e-1]:
            arms = [
                {'name': 'bp', 'args': '--method bp'},
                {'name': 'l2_init', 'args': '--method l2_init'},
                {'name': 'rs', 'args': '--method rs --lambda_rs 1.0'}
            ]
            for arm in arms:
                for seed in [1, 2, 3]:
                    cmd = f"PYTHONPATH=. python3 src/train.py --lr {lr} {arm['args']} --seed {seed} --n_tasks 150 --run_id S6_1_lr_{lr}_{arm['name']}_seed{seed}"
                    commands.append(cmd)

    elif args.suite == 'S6_2':
        for opt in ['sgd', 'adamw']:
            for seed in [1, 2, 3]:
                cmd = f"PYTHONPATH=. python3 src/train.py --optimizer {opt} --method rs --lambda_rs 1.0 --seed {seed} --n_tasks 150 --run_id S6_2_opt_{opt}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R2_B1':
        for lr in [0.01, 0.1]:
            for seed in [1, 2, 3]:
                cmd = f"PYTHONPATH=. python3 src/train.py --method bp --dataset permuted_mnist --n_tasks 300 --lr {lr} --seed {seed} --run_id R2_B1_lr_{lr}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R2_B2':
        for ep in [1, 3, 10]:
            for seed in [1, 2, 3]:
                cmd = f"PYTHONPATH=. python3 src/train.py --method bp --dataset permuted_mnist --n_tasks 50 --lr 0.01 --epochs {ep} --log_within_task --seed {seed} --run_id R2_B2_ep_{ep}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R2_B4':
        lambdas = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]
        for lam in lambdas:
            for seed in [1, 2, 3]:
                cmd = f"PYTHONPATH=. python3 src/train.py --method rs --lambda_rs {lam} --dataset permuted_mnist --n_tasks 50 --epochs 10 --lr 0.1 --seed {seed} --run_id R2_B4_lam_{lam}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R2_C1':
        # Curve A (the BP frontier)
        for lr in [0.003, 0.01, 0.03, 0.1, 0.3]:
            for seed in [1, 2, 3, 4, 5]:
                cmd = f"PYTHONPATH=. python3 src/train.py --method bp --dataset permuted_mnist --n_tasks 50 --epochs 10 --lr {lr} --seed {seed} --run_id R2_C1_A_bp_lr_{lr}_seed{seed}"
                commands.append(cmd)
                
        # Curve B (the RS curve)
        lambdas = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        for lam in lambdas:
            for seed in [1, 2, 3, 4, 5]:
                method = 'bp' if lam == 0.0 else 'rs'
                cmd = f"PYTHONPATH=. python3 src/train.py --method {method} --lambda_rs {lam} --dataset permuted_mnist --n_tasks 50 --epochs 10 --lr 0.1 --seed {seed} --run_id R2_C1_B_rs_lam_{lam}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R3_R1':
        lambdas = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 'inf']
        for lam in lambdas:
            for seed in [1, 2, 3, 4, 5]:
                if lam == 'inf':
                    cmd = f"PYTHONPATH=. python3 src/train.py --hard_projection --dataset permuted_mnist --n_tasks 150 --epochs 10 --lr 0.1 --track_drift --seed {seed} --run_id R3_R1_lam_inf_seed{seed}"
                else:
                    method = 'bp' if lam == 0.0 else 'rs'
                    cmd = f"PYTHONPATH=. python3 src/train.py --method {method} --lambda_rs {lam} --dataset permuted_mnist --n_tasks 150 --epochs 10 --lr 0.1 --track_drift --seed {seed} --run_id R3_R1_lam_{lam}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R3_R2':
        arms = [
            {'name': 'bp', 'args': '--method bp'},
            {'name': 'l2', 'args': '--method l2 --weight_decay 1e-3'},
            {'name': 'l2_init', 'args': '--method l2_init'},
            {'name': 'si', 'args': '--method si'},
            {'name': 'mas', 'args': '--method mas'},
            {'name': 'ln_l2', 'args': '--method ln_l2 --weight_decay 1e-4'},
            {'name': 'sp', 'args': '--method sp'},
            {'name': 'rs', 'args': '--method rs --lambda_rs 0.03'},
        ]
        for arm in arms:
            for seed in [1, 2, 3, 4, 5]:
                cmd = f"PYTHONPATH=. python3 src/train.py {arm['args']} --dataset permuted_mnist --n_tasks 150 --epochs 10 --lr 0.1 --seed {seed} --run_id R3_R2_{arm['name']}_seed{seed}"
                commands.append(cmd)
                
        for ewc_lam in [100, 1000, 10000]:
            for seed in [1, 2, 3, 4, 5]:
                cmd = f"PYTHONPATH=. python3 src/train.py --method ewc --ewc_lambda {ewc_lam} --dataset permuted_mnist --n_tasks 150 --epochs 10 --lr 0.1 --seed {seed} --run_id R3_R2_ewc_{ewc_lam}_seed{seed}"
                commands.append(cmd)
                cmd_rs = f"PYTHONPATH=. python3 src/train.py --method rs_ewc --lambda_rs 0.03 --ewc_lambda {ewc_lam} --dataset permuted_mnist --n_tasks 150 --epochs 10 --lr 0.1 --seed {seed} --run_id R3_R2_rs_ewc_{ewc_lam}_seed{seed}"
                commands.append(cmd_rs)

    elif args.suite == 'R3_R4':
        for w in [256, 512, 1000, 2048]:
            for lam in [0.0, 0.03]:
                for seed in [1, 2, 3]:
                    method = 'bp' if lam == 0.0 else 'rs'
                    cmd = f"PYTHONPATH=. python3 src/train.py --method {method} --lambda_rs {lam} --width {w} --dataset permuted_mnist --n_tasks 150 --epochs 10 --lr 0.1 --seed {seed} --run_id R3_R4_w_{w}_lam_{lam}_seed{seed}"
                    commands.append(cmd)

    elif args.suite == 'R3_R6':
        for opt in ['sgd', 'adamw', 'adam', 'sgd_momentum']:
            for lam in [0.0, 0.03]:
                for seed in [1, 2, 3, 4, 5]:
                    method = 'bp' if lam == 0.0 else 'rs'
                    # Run Adam/AdamW with a smaller learning rate since lr=0.1 or 0.01 is often too high
                    run_lr = 0.001 if 'adam' in opt else 0.01
                    cmd = f"PYTHONPATH=. python3 src/train.py --optimizer {opt} --method {method} --lambda_rs {lam} --dataset permuted_mnist --n_tasks 150 --epochs 10 --lr {run_lr} --seed {seed} --run_id R3_R6_opt_{opt}_lam_{lam}_seed{seed}"
                    commands.append(cmd)

    elif args.suite == 'R3_A1':
        for lr in [0.01, 0.1]:
            for seed in [1, 2, 3]:
                cmd = f"PYTHONPATH=. python3 src/train.py --method bp --dataset permuted_mnist --n_tasks 300 --epochs 10 --lr {lr} --seed {seed} --run_id R3_A1_lr_{lr}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R3_A2':
        lambdas = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]
        for lam in lambdas:
            for seed in [1, 2, 3]:
                cmd = f"PYTHONPATH=. python3 src/train.py --method rs --lambda_rs {lam} --dataset permuted_mnist --n_tasks 50 --epochs 10 --lr 0.1 --seed {seed} --run_id R3_A2_lam_{lam}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R3_C1_iso':
        for seed in [1, 2, 3, 4, 5]:
            cmd = f"PYTHONPATH=. python3 src/train.py --method bp_iso --dataset permuted_mnist --n_tasks 50 --epochs 10 --lr 0.1 --seed {seed} --run_id R3_C1_bp_iso_seed{seed}"
            commands.append(cmd)

    elif args.suite == 'R3_C1_matched':
        for lr in [0.015, 0.02, 0.025]:
            for seed in [1, 2, 3, 4, 5]:
                cmd = f"PYTHONPATH=. python3 src/train.py --method bp --dataset permuted_mnist --n_tasks 50 --epochs 10 --lr {lr} --seed {seed} --run_id R3_C1_bp_matched_lr_{lr}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R3_A1b':
        for ep in [3, 30]:
            for seed in [1, 2, 3]:
                cmd = f"PYTHONPATH=. python3 src/train.py --method bp --dataset permuted_mnist --n_tasks 300 --epochs {ep} --lr 0.01 --seed {seed} --run_id R3_A1b_ep_{ep}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R4_T1':
        lambdas = [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.006, 0.01, 0.02, 0.03]
        for lam in lambdas:
            for seed in [1, 2, 3, 4, 5]:
                method = 'bp' if lam == 0.0 else 'rs'
                cmd = f"PYTHONPATH=. python3 src/train.py --method {method} --lambda_rs {lam} --dataset permuted_mnist --n_tasks 150 --epochs 10 --lr 0.1 --track_drift --seed {seed} --run_id R4_T1_lam_{lam}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R4_T2c':
        # Need to ensure iso_shrinkage.json is generated first!
        for method in ['bp', 'bp_iso', 'rs']:
            lam = 0.01 if method == 'rs' else 0.0
            for seed in [1, 2, 3, 4, 5]:
                cmd = f"PYTHONPATH=. python3 src/train.py --method {method} --lambda_rs {lam} --dataset permuted_mnist --n_tasks 150 --epochs 10 --lr 0.1 --track_drift --seed {seed} --run_id R4_T2c_{method}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R4_T3a':
        for lam in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]:
            for seed in [1, 2, 3, 4, 5]:
                method = 'bp' if lam == 0.0 else 'rs'
                cmd = f"PYTHONPATH=. python3 src/train.py --dataset rotating_mnist --model mlp --depth 2 --width 256 --n_tasks 100 --lambda_rs {lam} --method {method} --epochs 1 --lr 0.01 --seed {seed} --run_id R4_T3a_rot_lam_{lam}_seed{seed}"
                commands.append(cmd)

    elif args.suite == 'R4_T3b':
        arms = [
            {'name': 'bp', 'args': '--method bp'},
            {'name': 'rs', 'args': '--method rs --lambda_rs 0.01'},
            {'name': 'ewc', 'args': '--method ewc --ewc_lambda 1000'},
            {'name': 'ln_l2', 'args': '--method ln_l2 --weight_decay 1e-4'}
        ]
        for arm in arms:
            for seed in [1, 2, 3]:
                cmd = f"PYTHONPATH=. python3 src/train.py --dataset split_cifar100 --model convnet --n_tasks 10 {arm['args']} --epochs 1 --lr 0.01 --seed {seed} --run_id R4_T3b_{arm['name']}_seed{seed}"
                commands.append(cmd)


    n_gpus = 2
    jobs_per_gpu = 5
    max_concurrent = n_gpus * jobs_per_gpu
    
    running_procs = []
    
    for i, cmd in enumerate(commands):
        run_id = cmd.split("--run_id ")[1].split(" ")[0]
        parquet_path = os.path.join('results', run_id, 'metrics.parquet')
        if os.path.exists(parquet_path):
            print(f"Skipping {run_id}, already completed.")
            continue

        gpu_id = len(running_procs) % n_gpus
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        while len(running_procs) >= max_concurrent:
            time.sleep(1)
            running_procs = [(p, cmd_str) for p, cmd_str in running_procs if p.poll() is None]
            
        print(f"Launching on GPU {gpu_id}: {cmd}")
        # redirecting to avoid terminal clutter, but writing to individual logs
        log_file = cmd.split("--run_id ")[1].split(" ")[0] + ".log"
        log_path = os.path.join('results', log_file)
        os.makedirs('results', exist_ok=True)
        
        with open(log_path, 'w') as f:
            p = subprocess.Popen(cmd, shell=True, env=env, stdout=f, stderr=f)
        running_procs.append((p, cmd))
        
        # small delay to prevent rapid memory spikes during init
        time.sleep(2)
        
    while len(running_procs) > 0:
        time.sleep(1)
        running_procs = [(p, cmd_str) for p, cmd_str in running_procs if p.poll() is None]
        
    print(f"Sweep {args.suite} complete.")

if __name__ == '__main__':
    main()
