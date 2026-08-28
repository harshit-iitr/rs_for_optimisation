import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_c1():
    base_dir = 'results'
    all_data = []
    
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R2_C1_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        
        df = pd.read_parquet(path)
        if 'layer' in df.columns:
            df = df[df['layer'] == 0]
            
        if '_bp_' in run_id:
            lr = float(run_id.split('lr_')[1].split('_')[0])
            arm_name = f"BP (lr={lr})"
            curve = 'A'
            x_val = lr
        else:
            lam = float(run_id.split('lam_')[1].split('_')[0])
            if lam == 0.0:
                arm_name = "BP (lr=0.1)"
            else:
                arm_name = f"RS (lam={lam})"
            curve = 'B'
            x_val = lam
            
        df['arm_name'] = arm_name
        df['curve'] = curve
        df['x_val'] = x_val
        all_data.append(df)
        
    if not all_data:
        print("No C1 data found.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    
    # Average over final 20 tasks (tasks 30 to 49)
    df_final = df[df['task'] >= 30]
    
    # Mean over tasks for each run
    run_means = df_final.groupby(['arm_name', 'curve', 'seed', 'x_val'])[['test_acc', 'prev_tasks_acc']].mean().reset_index()
    
    # Mean and std over seeds
    final_stats = run_means.groupby(['arm_name', 'curve', 'x_val'])[['test_acc', 'prev_tasks_acc']].agg(['mean', 'std']).reset_index()
    final_stats.columns = ['arm_name', 'curve', 'x_val', 'test_acc_mean', 'test_acc_std', 'prev_tasks_acc_mean', 'prev_tasks_acc_std']
    
    plt.figure(figsize=(10, 8))
    
    curve_a = final_stats[final_stats['curve'] == 'A'].sort_values('x_val')
    plt.errorbar(curve_a['test_acc_mean'], curve_a['prev_tasks_acc_mean'],
                 xerr=curve_a['test_acc_std'], yerr=curve_a['prev_tasks_acc_std'],
                 fmt='-o', label='Curve A (BP Frontier, var LR)', capsize=3)
                 
    curve_b = final_stats[final_stats['curve'] == 'B'].sort_values('x_val')
    plt.errorbar(curve_b['test_acc_mean'], curve_b['prev_tasks_acc_mean'],
                 xerr=curve_b['test_acc_std'], yerr=curve_b['prev_tasks_acc_std'],
                 fmt='-s', label='Curve B (RS, fixed LR=0.1, var lam)', capsize=3)
                 
    for _, row in curve_a.iterrows():
        plt.annotate(row['arm_name'], (row['test_acc_mean'], row['prev_tasks_acc_mean']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    for _, row in curve_b.iterrows():
        plt.annotate(row['arm_name'], (row['test_acc_mean'], row['prev_tasks_acc_mean']),
                     textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
                     
    plt.xlabel('Plasticity (Current Task Accuracy, final 20 tasks)')
    plt.ylabel('Stability (Previous Tasks Retention, final 20 tasks)')
    plt.title('C1: The Critical Control (Pareto Frontier)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/c1_frontier.png')
    print("Saved plot to results/c1_frontier.png")
    
    # Also save to CSV to include in report
    final_stats.to_csv('results/c1_stats.csv', index=False)

if __name__ == '__main__':
    analyze_c1()
