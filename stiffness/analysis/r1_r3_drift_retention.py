import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_r1_r3():
    base_dir = 'results'
    all_data = []
    
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R3_R1_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        all_data.append(df)
        
    if not all_data:
        print("No R1/R3 data found.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    df_layer0 = df[df['layer'] == 0]
    
    # Average over final 20 tasks
    df_final = df_layer0[df_layer0['task'] >= 130]
    
    run_means = df_final.groupby(['lambda_rs', 'hard_projection', 'seed'])[['test_acc', 'prev_tasks_acc', 'drift_rel']].mean().reset_index()
    stats = run_means.groupby(['lambda_rs', 'hard_projection'])[['test_acc', 'prev_tasks_acc', 'drift_rel']].agg(['mean', 'std']).reset_index()
    stats.columns = ['lambda_rs', 'hard_projection', 'test_acc_mean', 'test_acc_std', 'prev_tasks_acc_mean', 'prev_tasks_acc_std', 'drift_rel_mean', 'drift_rel_std']
    
    # Separate finite lambdas vs hard projection
    finite = stats[~stats['hard_projection']].sort_values('lambda_rs')
    hard = stats[stats['hard_projection']].iloc[0]
    
    finite = finite[finite['lambda_rs'] > 0] # only plot > 0 on log scale
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Retention vs Lambda
    axes[0].errorbar(finite['lambda_rs'], finite['prev_tasks_acc_mean'], yerr=finite['prev_tasks_acc_std'], fmt='-o')
    axes[0].axhline(hard['prev_tasks_acc_mean'], color='r', linestyle='--', label='Hard Projection')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Lambda')
    axes[0].set_ylabel('Retention')
    axes[0].set_title('R1: Retention vs Lambda')
    axes[0].legend()
    
    # 2. Drift vs Lambda
    axes[1].errorbar(finite['lambda_rs'], finite['drift_rel_mean'], yerr=finite['drift_rel_std'], fmt='-o')
    axes[1].axhline(hard['drift_rel_mean'], color='r', linestyle='--', label='Hard Projection')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Lambda')
    axes[1].set_ylabel('Relative Drift')
    axes[1].set_title('R3: Relative Drift vs Lambda (log-log)')
    axes[1].legend()
    
    # 3. Retention vs Drift
    axes[2].errorbar(finite['drift_rel_mean'], finite['prev_tasks_acc_mean'], 
                     xerr=finite['drift_rel_std'], yerr=finite['prev_tasks_acc_std'], fmt='o')
    axes[2].plot(hard['drift_rel_mean'], hard['prev_tasks_acc_mean'], 'rs', label='Hard Projection')
    axes[2].set_xlabel('Relative Drift')
    axes[2].set_ylabel('Retention')
    axes[2].set_title('R1/R3: Retention vs Drift')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('results/r1_r3_drift_retention.png')
    print("Saved plot to results/r1_r3_drift_retention.png")
    
    stats.to_csv('results/r1_r3_stats.csv', index=False)
    print("Does hard projection beat best lambda?")
    best_finite = finite.loc[finite['prev_tasks_acc_mean'].idxmax()]
    print(f"Best Finite: lam={best_finite['lambda_rs']}, Ret={best_finite['prev_tasks_acc_mean']:.4f}")
    print(f"Hard Proj: Ret={hard['prev_tasks_acc_mean']:.4f}")

if __name__ == '__main__':
    analyze_r1_r3()
