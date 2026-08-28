import pandas as pd
import numpy as np
import os
from scipy import stats

def main():
    seeds = [1, 2, 3, 4, 5]
    arms = [
        'bp', 'l2_1e-3', 'l2_1e-4', 'ln', 'ln_l2', 'sp', 'l2_init', 
        'cbp', 'redo', 'er', 'l2_er', 'rs', 'rs_cbp'
    ]
    
    records = []
    
    for arm in arms:
        for seed in seeds:
            run_id = f"S2_{arm}_seed{seed}"
            path = f"results/{run_id}/metrics.parquet"
            if not os.path.exists(path):
                continue
                
            df = pd.read_parquet(path)
            
            task_df = df.groupby('task').mean(numeric_only=True)
            
            final_acc = task_df.loc[130:149, 'test_acc'].mean() if 149 in task_df.index else np.nan
            peak_dead = task_df['dead_frac'].max()
            
            plast_loss = task_df.loc[0, 'test_acc'] - task_df.loc[149, 'test_acc'] if 149 in task_df.index and 0 in task_df.index else np.nan
            
            records.append({
                'arm': arm,
                'seed': seed,
                'final_acc': final_acc,
                'peak_dead_frac': peak_dead,
                'plasticity_loss': plast_loss
            })
            
    if not records:
        print("No S2 records found.")
        return
        
    res_df = pd.DataFrame(records)
    
    summary = res_df.groupby('arm').agg({
        'final_acc': ['mean', 'std'],
        'peak_dead_frac': ['mean', 'std'],
        'plasticity_loss': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['arm', 'acc_mean', 'acc_std', 'dead_mean', 'dead_std', 'plast_mean', 'plast_std']
    summary = summary.sort_values('acc_mean', ascending=False)
    
    print(f"{'Method':<15} | {'Final Acc (±std)':<20} | {'Peak Dead (±std)':<20} | {'Plast Loss (±std)':<20}")
    print("-" * 80)
    for _, row in summary.iterrows():
        print(f"{row['arm']:<15} | {row['acc_mean']:.4f} ± {row['acc_std']:.4f} | {row['dead_mean']:.4f} ± {row['dead_std']:.4f} | {row['plast_mean']:.4f} ± {row['plast_std']:.4f}")
        
    print("\n--- Significance Tests ---")
    rs_vals = res_df[res_df['arm'] == 'rs']['final_acc'].values
    if len(rs_vals) == 0:
        return
        
    baselines = summary[~summary['arm'].str.startswith('rs')]
    if len(baselines) > 0:
        best_base = baselines.iloc[0]['arm']
        base_vals = res_df[res_df['arm'] == best_base]['final_acc'].values
        
        if len(rs_vals) == len(base_vals):
            t, p = stats.ttest_rel(rs_vals, base_vals)
            print(f"RS vs {best_base} (paired): t={t:.3f}, p={p:.4f}")
        else:
            t, p = stats.ttest_ind(rs_vals, base_vals)
            print(f"RS vs {best_base} (indep): t={t:.3f}, p={p:.4f}")
            
    l2_1e3 = res_df[res_df['arm'] == 'l2_1e-3']['final_acc'].values
    l2_1e4 = res_df[res_df['arm'] == 'l2_1e-4']['final_acc'].values
    
    if len(l2_1e3) > 0 and len(l2_1e4) > 0:
        l2_best = 'l2_1e-3' if l2_1e3.mean() > l2_1e4.mean() else 'l2_1e-4'
        l2_vals = res_df[res_df['arm'] == l2_best]['final_acc'].values
        if len(rs_vals) == len(l2_vals):
            t, p = stats.ttest_rel(rs_vals, l2_vals)
            print(f"RS vs {l2_best} (paired): t={t:.3f}, p={p:.4f}")
            if p > 0.05 or rs_vals.mean() <= l2_vals.mean():
                print("\n[!!!] KILL CONDITION TRIGGERED: RS does not clearly beat standard L2.")

if __name__ == '__main__':
    main()
