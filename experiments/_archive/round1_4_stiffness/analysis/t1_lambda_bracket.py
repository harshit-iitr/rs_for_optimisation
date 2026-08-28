import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_t1():
    base_dir = 'results'
    all_data = []
    
    # Load R3_R1 runs (the original lambdas)
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R3_R1_lam_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        parts = run_id.split('_')
        lam_str = parts[3]
        lam = float('inf') if lam_str == 'inf' else float(lam_str)
        df['lam'] = lam
        df['suite'] = 'R3_R1'
        all_data.append(df)
        
    # Load R4_T1 runs (the fine-grained lambdas)
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R4_T1_lam_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        parts = run_id.split('_')
        lam_str = parts[3]
        lam = float('inf') if lam_str == 'inf' else float(lam_str)
        df['lam'] = lam
        df['suite'] = 'R4_T1'
        all_data.append(df)
        
    if not all_data:
        print("No data found for R3_R1 or R4_T1.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    df = df[df['layer'] == 0] # Use first layer for representation metrics
    
    # Filter for final 20 tasks
    df_final = df[df['task'] >= 130]
    
    # Aggregate over tasks and seeds
    stats = df_final.groupby(['lam', 'seed'])[['test_acc', 'prev_tasks_acc', 'drift_overlap', 'drift_rad', 'radial_excess']].mean().reset_index()
    
    final_stats = stats.groupby('lam').agg(['mean', 'std', 'count']).reset_index()
    
    # Flatten columns
    final_stats.columns = ['lam'] + [f'{col}_{stat}' for col, stat in final_stats.columns[1:]]
    
    # Save table
    print("=== T1: Lambda Bracket ===")
    print(final_stats.to_string(index=False))
    final_stats.to_csv('results/t1_lambda_bracket.csv', index=False)
    
    # Plot Current-Task Acc vs Retention with error bars
    plt.figure(figsize=(10, 6))
    finite = final_stats[final_stats['lam'] != float('inf')]
    finite = finite[finite['lam'] >= 0.0]
    finite = finite.sort_values('lam')
    
    plt.errorbar(finite['lam'], finite['prev_tasks_acc_mean'], yerr=finite['prev_tasks_acc_std'], fmt='-o', color='blue', label='Retention (prev_tasks_acc)')
    plt.errorbar(finite['lam'], finite['test_acc_mean'], yerr=finite['test_acc_std'], fmt='-s', color='orange', label='Current-Task Acc (test_acc)')
    
    if float('inf') in final_stats['lam'].values:
        inf_val_ret = final_stats[final_stats['lam'] == float('inf')]['prev_tasks_acc_mean'].values[0]
        inf_val_cur = final_stats[final_stats['lam'] == float('inf')]['test_acc_mean'].values[0]
        plt.axhline(inf_val_ret, color='blue', linestyle='--', label='Hard Projection (Retention)')
        plt.axhline(inf_val_cur, color='orange', linestyle='--', label='Hard Projection (Current Acc)')
        
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('T1: Accuracy & Retention vs Lambda (Permuted MNIST, 150 tasks)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/t1_lambda_bracket.png')

if __name__ == '__main__':
    analyze_t1()
