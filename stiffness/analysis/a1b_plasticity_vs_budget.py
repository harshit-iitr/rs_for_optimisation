import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def analyze_a1b():
    base_dir = 'results'
    all_data = []
    
    # epochs=1 from R2_B1
    for s in [1, 2, 3]:
        path = f'{base_dir}/R2_B1_lr_0.01_seed{s}/metrics.parquet'
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['epochs'] = 1
            all_data.append(df)
            
    # epochs=3 from R3_A1b
    for s in [1, 2, 3]:
        path = f'{base_dir}/R3_A1b_ep_3_seed{s}/metrics.parquet'
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['epochs'] = 3
            all_data.append(df)
            
    # epochs=10 from R3_A1
    for s in [1, 2, 3]:
        path = f'{base_dir}/R3_A1_lr_0.01_seed{s}/metrics.parquet'
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['epochs'] = 10
            all_data.append(df)
            
    # epochs=30 from R3_A1b
    for s in [1, 2, 3]:
        path = f'{base_dir}/R3_A1b_ep_30_seed{s}/metrics.parquet'
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['epochs'] = 30
            all_data.append(df)
            
    if not all_data: return
    df = pd.concat(all_data, ignore_index=True)
    df_l0 = df[df['layer'] == 0]
    
    results = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ep in sorted(df['epochs'].unique()):
        sub = df_l0[df_l0['epochs'] == ep]
        mean_traj = sub.groupby('task')[['test_acc', 'weight_norm', 'dead_frac']].mean()
        
        axes[1].plot(mean_traj.index, mean_traj['weight_norm'], label=f'{ep} epochs')
        axes[2].plot(mean_traj.index, mean_traj['dead_frac'], label=f'{ep} epochs')
        
        sub_fit = sub[(sub['task'] >= 50) & (sub['task'] <= 300)]
        res = stats.linregress(sub_fit['task'], sub_fit['test_acc'])
        
        results.append({
            'epochs': ep,
            'slope': res.slope,
            'stderr': res.stderr,
            'p_value': res.pvalue
        })
        
    res_df = pd.DataFrame(results)
    print("=== A1b: Plasticity Loss vs Budget ===")
    print(res_df.to_string(index=False))
    
    axes[0].errorbar(res_df['epochs'], res_df['slope'], yerr=res_df['stderr']*1.96, fmt='-o')
    axes[0].set_xlabel('Epochs per Task')
    axes[0].set_ylabel('Accuracy Slope (Tasks 50-300)')
    axes[0].set_title('Plasticity Loss Slope vs Budget')
    
    axes[1].set_xlabel('Task')
    axes[1].set_ylabel('Weight Norm (L0)')
    axes[1].set_title('Weight Norm Trajectories')
    axes[1].legend()
    
    axes[2].set_xlabel('Task')
    axes[2].set_ylabel('Dead Fraction (L0)')
    axes[2].set_title('Dead Fraction Trajectories')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('results/a1b_plasticity_vs_budget.png')
    res_df.to_csv('results/a1b_stats.csv', index=False)

if __name__ == '__main__':
    analyze_a1b()
