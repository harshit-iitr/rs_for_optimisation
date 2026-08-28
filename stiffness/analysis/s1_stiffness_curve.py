import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import stats

def main():
    lambdas = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 'inf']
    seeds = [1, 2, 3]
    
    records = []
    
    for lam in lambdas:
        for seed in seeds:
            run_id = f"S1_lam_{lam}_seed{seed}"
            path = f"results/{run_id}/metrics.parquet"
            if not os.path.exists(path):
                continue
                
            df = pd.read_parquet(path)
            # mean final-20-task accuracy:
            final_20 = df[df['task'] >= 130] # 130 to 149
            
            # average over tasks for the single seed
            # Since df has one row per (task, layer), we group by task first
            task_df = final_20.groupby('task').mean(numeric_only=True)
            
            acc = task_df['test_acc'].mean()
            dead_frac = task_df['dead_frac'].mean()
            eff_rank = task_df['eff_rank'].mean()
            phi_rad = task_df['phi_rad_tilde'].mean()
            
            records.append({
                'lambda': lam if lam != 'inf' else 100.0, # proxy for inf in plot
                'is_inf': lam == 'inf',
                'seed': seed,
                'test_acc': acc,
                'dead_frac': dead_frac,
                'eff_rank': eff_rank,
                'phi_rad_tilde': phi_rad
            })
            
    if not records:
        print("No records found.")
        return
        
    res_df = pd.DataFrame(records)
    
    summary = res_df.groupby(['lambda', 'is_inf']).agg(['mean', 'std']).reset_index()
    
    metrics = ['test_acc', 'dead_frac', 'eff_rank', 'phi_rad_tilde']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, m in enumerate(metrics):
        ax = axes[i]
        
        # Soft lambda points
        soft = summary[~summary['is_inf']]
        xs = soft['lambda']
        ys = soft[m]['mean']
        yerr = soft[m]['std']
        
        ax.errorbar(xs, ys, yerr=yerr, fmt='-o', label='Soft RS')
        
        # Hard projection
        hard = summary[summary['is_inf']]
        if len(hard) > 0:
            ax.axhline(hard[m]['mean'].values[0], color='r', linestyle='--', label='Hard Proj (inf)')
            
        ax.set_xscale('symlog', linthresh=1e-4)
        ax.set_xlabel('Lambda')
        ax.set_ylabel(m)
        ax.set_title(f'Stiffness Curve: {m}')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig('results/s1_stiffness_curve.png')
    print("Saved S1 curve to results/s1_stiffness_curve.png")
    
    # Also find optimal interior lambda
    soft_res = res_df[~res_df['is_inf']].groupby('lambda')['test_acc'].mean()
    best_lam = soft_res.idxmax()
    print(f"\nBest interior lambda: {best_lam} (Acc: {soft_res.max():.4f})")
    
    # Check if interior beats both ERM (0.0) and Hard (inf)
    erm_acc = soft_res.loc[0.0] if 0.0 in soft_res.index else np.nan
    hard_acc = res_df[res_df['is_inf']]['test_acc'].mean() if len(res_df[res_df['is_inf']]) > 0 else np.nan
    print(f"ERM Acc (λ=0): {erm_acc:.4f}")
    print(f"Hard Proj Acc (λ=inf): {hard_acc:.4f}")
    
    # Significance test (paired t-test over seeds)
    if 0.0 in soft_res.index and best_lam != 0.0:
        best_vals = res_df[(res_df['lambda'] == best_lam) & (~res_df['is_inf'])]['test_acc'].values
        erm_vals = res_df[(res_df['lambda'] == 0.0) & (~res_df['is_inf'])]['test_acc'].values
        t, p = stats.ttest_rel(best_vals, erm_vals)
        print(f"Best vs ERM: paired t={t:.3f}, p={p:.4f}, n={len(best_vals)}")

if __name__ == '__main__':
    main()
