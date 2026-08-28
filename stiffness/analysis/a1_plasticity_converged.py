import os
import pandas as pd
import numpy as np
from scipy.stats import linregress

def analyze_a1():
    base_dir = 'results'
    all_data = []
    
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R3_A1_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        lr = float(run_id.split('_lr_')[1].split('_seed')[0])
        df['lr'] = lr
        all_data.append(df)
        
    if not all_data:
        print("No A1 data found.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    df_layer0 = df[df['layer'] == 0]
    
    results = []
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for lr in df['lr'].unique():
        sub = df_layer0[df_layer0['lr'] == lr]
        
        mean_traj = sub.groupby('task')[['test_acc', 'weight_norm', 'dead_frac']].mean()
        
        axes[0].plot(mean_traj.index, mean_traj['test_acc'], label=f'lr={lr}')
        axes[1].plot(mean_traj.index, mean_traj['weight_norm'], label=f'lr={lr}')
        axes[2].plot(mean_traj.index, mean_traj['dead_frac'], label=f'lr={lr}')
        
        sub_fit = sub[(sub['task'] >= 50) & (sub['task'] <= 300)]
        
        res = linregress(sub_fit['task'], sub_fit['test_acc'])
        
        results.append({
            'lr': lr,
            'slope': res.slope,
            'p_value': res.pvalue,
            'stderr': res.stderr
        })
        
    axes[0].set_title('Test Accuracy vs Task')
    axes[1].set_title('Weight Norm vs Task (Layer 0)')
    axes[2].set_title('Dead Frac vs Task (Layer 0)')
    axes[0].legend()
    plt.tight_layout()
    plt.savefig('results/a1_trajectories.png')
    
    res_df = pd.DataFrame(results)
    print("=== A1: Plasticity Loss at Converged Budget (Tasks 50-300) ===")
    print(res_df.to_string(index=False))
    res_df.to_csv('results/a1_stats.csv', index=False)

if __name__ == '__main__':
    analyze_a1()
