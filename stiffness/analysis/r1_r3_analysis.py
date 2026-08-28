import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import stats

def main():
    lambdas = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    
    records = []
    
    # We will gather R3_R1_lam_* and R3_R1_lam_inf_*
    for lam in lambdas:
        for seed in range(1, 6):
            path = f"results/R3_R1_lam_{lam}_seed{seed}/metrics.parquet"
            if not os.path.exists(path):
                # Try 0.0 instead of 0 if missing
                if lam == 0:
                    path = f"results/R3_R1_lam_0.0_seed{seed}/metrics.parquet"
            if os.path.exists(path):
                try:
                    df = pd.read_parquet(path)
                    # Average over tasks 130-149
                    task_df = df.groupby('task').mean(numeric_only=True)
                    fin_acc = task_df.loc[130:149, 'test_acc'].mean()
                    retention = task_df.loc[130:149, 'prev_tasks_acc'].mean()
                    
                    drift_cos = task_df.loc[130:149, 'drift_cos_sim'].mean()
                    drift_rel = task_df.loc[130:149, 'drift_rel'].mean()
                    drift_overlap = task_df.loc[130:149, 'drift_overlap'].mean()
                    
                    records.append({
                        'lambda_rs': lam,
                        'seed': seed,
                        'test_acc': fin_acc,
                        'retention': retention,
                        'drift_cos': drift_cos,
                        'drift_rel': drift_rel,
                        'drift_overlap': drift_overlap
                    })
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    
    # Also read inf
    for seed in range(1, 6):
        path = f"results/R3_R1_lam_inf_seed{seed}/metrics.parquet"
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                task_df = df.groupby('task').mean(numeric_only=True)
                fin_acc = task_df.loc[130:149, 'test_acc'].mean()
                retention = task_df.loc[130:149, 'prev_tasks_acc'].mean()
                drift_cos = task_df.loc[130:149, 'drift_cos_sim'].mean()
                drift_rel = task_df.loc[130:149, 'drift_rel'].mean()
                drift_overlap = task_df.loc[130:149, 'drift_overlap'].mean()
                records.append({
                    'lambda_rs': np.inf,
                    'seed': seed,
                    'test_acc': fin_acc,
                    'retention': retention,
                    'drift_cos': drift_cos,
                    'drift_rel': drift_rel,
                    'drift_overlap': drift_overlap
                })
            except Exception:
                pass
                
    res_df = pd.DataFrame(records)
    
    if len(res_df) == 0:
        print("No records found for R1/R3.")
        return
        
    agg = res_df.groupby('lambda_rs').agg(['mean', 'std']).reset_index()
    
    print("--- R1 Retention Sweep ---")
    for _, row in agg.iterrows():
        lam = row[('lambda_rs', '')]
        acc_mean = row[('test_acc', 'mean')]
        ret_mean = row[('retention', 'mean')]
        print(f"λ={lam}: Acc={acc_mean:.4f}, Ret={ret_mean:.4f}")
        
    print("\n--- R3 Drift Metrics ---")
    for _, row in agg.iterrows():
        lam = row[('lambda_rs', '')]
        cos_mean = row[('drift_cos', 'mean')]
        rel_mean = row[('drift_rel', 'mean')]
        overlap = row[('drift_overlap', 'mean')]
        print(f"λ={lam}: Cos={cos_mean:.4f}, RelDrift={rel_mean:.4f}, Overlap={overlap:.4f}")
        
    # Plot retention
    plt.figure(figsize=(10, 5))
    
    # Filter out inf for x-axis scale plotting (or plot it separately)
    finite_agg = agg[agg[('lambda_rs', '')] != np.inf]
    x = finite_agg[('lambda_rs', '')]
    
    plt.subplot(1, 2, 1)
    plt.errorbar(x, finite_agg[('test_acc', 'mean')], yerr=finite_agg[('test_acc', 'std')], label='Current Acc', marker='o')
    plt.errorbar(x, finite_agg[('retention', 'mean')], yerr=finite_agg[('retention', 'std')], label='Retention', marker='s')
    plt.xscale('symlog', linthresh=0.01)
    plt.title('R1: Accuracy & Retention vs λ')
    plt.xlabel('λ')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.errorbar(x, finite_agg[('drift_cos', 'mean')], yerr=finite_agg[('drift_cos', 'std')], label='Cosine Sim', marker='^')
    plt.errorbar(x, finite_agg[('drift_overlap', 'mean')], yerr=finite_agg[('drift_overlap', 'std')], label='Subspace Overlap', marker='v')
    plt.xscale('symlog', linthresh=0.01)
    plt.title('R3: Drift Metrics vs λ')
    plt.xlabel('λ')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/r1_r3_analysis.png')
    print("Saved plots to results/r1_r3_analysis.png")
    
if __name__ == '__main__':
    main()
