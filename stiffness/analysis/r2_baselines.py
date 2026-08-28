import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_r2():
    base_dir = 'results'
    all_data = []
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R3_R2_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        parts = run_id.replace('R3_R2_', '').split('_seed')
        arm_name = parts[0]
        df['arm_name'] = arm_name
        all_data.append(df)
        
    if not all_data: return
    df = pd.concat(all_data, ignore_index=True)
    df = df[df['layer'] == 0]
    
    df_final = df[df['task'] >= 130]
    stats = df_final.groupby(['arm_name', 'seed'])[['test_acc', 'prev_tasks_acc']].mean().reset_index()
    final_stats = stats.groupby('arm_name')[['test_acc', 'prev_tasks_acc']].agg(['mean', 'std']).reset_index()
    final_stats.columns = ['arm_name', 'test_acc_mean', 'test_acc_std', 'prev_tasks_acc_mean', 'prev_tasks_acc_std']
    
    ewc_arms = final_stats[final_stats['arm_name'].str.startswith('ewc_')]
    if not ewc_arms.empty:
        best_ewc = ewc_arms.loc[ewc_arms['prev_tasks_acc_mean'].idxmax()]
        final_stats = final_stats[~final_stats['arm_name'].str.startswith('ewc_')]
        final_stats = pd.concat([final_stats, best_ewc.to_frame().T])
        
    rs_ewc_arms = final_stats[final_stats['arm_name'].str.startswith('rs_ewc_')]
    if not rs_ewc_arms.empty:
        best_rs_ewc = rs_ewc_arms.loc[rs_ewc_arms['prev_tasks_acc_mean'].idxmax()]
        final_stats = final_stats[~final_stats['arm_name'].str.startswith('rs_ewc_')]
        final_stats = pd.concat([final_stats, best_rs_ewc.to_frame().T])
    
    plt.figure(figsize=(12, 10))
    for _, row in final_stats.iterrows():
        plt.errorbar(row['test_acc_mean'], row['prev_tasks_acc_mean'], 
                     xerr=row['test_acc_std'], yerr=row['prev_tasks_acc_std'], fmt='o', label=row['arm_name'])
        plt.annotate(row['arm_name'], (row['test_acc_mean'], row['prev_tasks_acc_mean']), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    plt.xlabel('Current Task Accuracy')
    plt.ylabel('Previous Tasks Retention')
    plt.title('R2: Stability Baselines (Pareto Frontier)')
    plt.grid(True)
    plt.savefig('results/r2_baselines.png')
    
    final_stats.to_csv('results/r2_stats.csv', index=False)

if __name__ == '__main__':
    analyze_r2()
