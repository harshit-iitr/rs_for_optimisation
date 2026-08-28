import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def analyze_c1():
    base_dir = 'results'
    all_data = []
    
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R2_C1_A_bp_lr_') and not run_id.startswith('R3_C1_bp_matched_lr_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        lr = float(run_id.split('_lr_')[1].split('_seed')[0])
        df['lr'] = lr
        df['arm'] = 'BP'
        all_data.append(df)
        
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R2_C1_B_rs_lam_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        lam = float(run_id.split('_lam_')[1].split('_seed')[0])
        df['lam'] = lam
        df['arm'] = 'BP' if lam == 0.0 else 'RS'
        all_data.append(df)
        
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R3_C1_bp_iso_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        df['arm'] = 'BP_ISO'
        all_data.append(df)
        
    if not all_data: return
    df = pd.concat(all_data, ignore_index=True)
    df = df[df['layer'] == 0]
    
    df_final = df[df['task'] >= 30]
    
    df_bp = df_final[df_final['arm'] == 'BP']
    bp_stats = df_bp.groupby(['lr', 'seed'])[['test_acc', 'prev_tasks_acc', 'update_norm']].mean().reset_index()
    bp_final = bp_stats.groupby('lr')[['test_acc', 'prev_tasks_acc', 'update_norm']].agg(['mean', 'std']).reset_index()
    
    df_rs = df_final[df_final['arm'] == 'RS']
    rs_stats = df_rs.groupby(['lam', 'seed'])[['test_acc', 'prev_tasks_acc', 'update_norm']].mean().reset_index()
    rs_final = rs_stats.groupby('lam')[['test_acc', 'prev_tasks_acc', 'update_norm']].agg(['mean', 'std']).reset_index()
    
    df_iso = df_final[df_final['arm'] == 'BP_ISO']
    
    print("=== BP Stats ===")
    print(bp_final)
    print("=== RS Stats ===")
    print(rs_final)
    
    rs_best = rs_stats[rs_stats['lam'] == 0.03]
    rs_best_acc = rs_best['test_acc'].mean()
    rs_best_ret = rs_best['prev_tasks_acc'].mean()
    
    bp_mean = bp_stats.groupby('lr')['test_acc'].mean()
    best_lr = (bp_mean - rs_best_acc).abs().idxmin()
    bp_match = bp_stats[bp_stats['lr'] == best_lr]
    
    print(f"\nRS Best (lam=0.03): Acc={rs_best_acc:.4f}, Ret={rs_best_ret:.4f}")
    print(f"BP Match (lr={best_lr}): Acc={bp_match['test_acc'].mean():.4f}, Ret={bp_match['prev_tasks_acc'].mean():.4f}")
    
    t_stat, p_val = stats.ttest_ind(rs_best['prev_tasks_acc'].values, bp_match['prev_tasks_acc'].values)
    print(f"Direct Match T-test: t={t_stat:.2f}, p={p_val:.4e}")
    
    if not df_iso.empty:
        iso_stats = df_iso.groupby('seed')[['test_acc', 'prev_tasks_acc', 'update_norm']].mean().reset_index()
        iso_final = iso_stats[['test_acc', 'prev_tasks_acc', 'update_norm']].agg(['mean', 'std']).T
        print("\n=== ISO Stats ===")
        print(iso_final)
        
        print(f"\nBP ISO: Acc={iso_stats['test_acc'].mean():.4f}, Ret={iso_stats['prev_tasks_acc'].mean():.4f}")
        t_stat_iso, p_val_iso = stats.ttest_ind(rs_best['prev_tasks_acc'].values, iso_stats['prev_tasks_acc'].values)
        print(f"RS vs BP_ISO T-test: t={t_stat_iso:.2f}, p={p_val_iso:.4e}")
        
    plt.figure(figsize=(10, 8))
    plt.errorbar(bp_final['test_acc']['mean'], bp_final['prev_tasks_acc']['mean'], 
                 xerr=bp_final['test_acc']['std'], yerr=bp_final['prev_tasks_acc']['std'], 
                 fmt='-o', label='BP (Curve A)')
                 
    plt.errorbar(rs_final['test_acc']['mean'], rs_final['prev_tasks_acc']['mean'], 
                 xerr=rs_final['test_acc']['std'], yerr=rs_final['prev_tasks_acc']['std'], 
                 fmt='-o', label='RS (Curve B)')
                 
    if not df_iso.empty:
        plt.errorbar(iso_stats['test_acc'].mean(), iso_stats['prev_tasks_acc'].mean(),
                     xerr=iso_stats['test_acc'].std(), yerr=iso_stats['prev_tasks_acc'].std(),
                     fmt='s', markersize=10, label='BP Isotropic Control')
                     
    plt.xlabel('Current Task Accuracy')
    plt.ylabel('Previous Tasks Retention')
    plt.title('C1: The Plasticity-Stability Frontier')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/c1_frontier_v2.png')

if __name__ == '__main__':
    analyze_c1()
