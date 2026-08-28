import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def analyze_b1():
    base_dir = 'results'
    all_data = []
    
    # Read metrics.parquet for B1 runs
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R2_B1_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        
        lr = float(run_id.split('lr_')[1].split('_')[0])
        seed = int(run_id.split('seed')[-1])
        
        df = pd.read_parquet(path)
        if 'layer' in df.columns:
            df = df[df['layer'] == 0]
            
        df['lr'] = lr
        df['seed'] = seed
        all_data.append(df)
        
    if not all_data:
        print("No B1 data found.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    
    plt.figure(figsize=(10, 6))
    for lr in sorted(df['lr'].unique()):
        df_lr = df[df['lr'] == lr]
        
        # Mean and std across seeds
        grouped = df_lr.groupby('task')['test_acc'].agg(['mean', 'std']).reset_index()
        
        plt.plot(grouped['task'], grouped['mean'], label=f'LR = {lr}')
        plt.fill_between(grouped['task'], grouped['mean'] - grouped['std'], grouped['mean'] + grouped['std'], alpha=0.2)
        
        # Fit linear slope over tasks 50-300
        df_fit = df_lr[df_lr['task'] >= 50]
        if len(df_fit) > 0:
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_fit['task'], df_fit['test_acc'])
            ci = 1.96 * std_err
            print(f"LR {lr}: Degradation slope over tasks 50-300 = {slope:.2e} ± {ci:.2e} (p={p_value:.2e})")
            
    plt.xlabel('Task')
    plt.ylabel('Test Accuracy')
    plt.title('B1: Plasticity Loss (Test Accuracy vs Task)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/b1_plasticity_loss.png')
    print("Saved plot to results/b1_plasticity_loss.png")

if __name__ == '__main__':
    analyze_b1()
