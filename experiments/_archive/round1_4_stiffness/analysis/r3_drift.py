import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def analyze_r3():
    base_dir = 'results'
    all_data = []
    
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R3_R1_lam_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        lam_str = run_id.split('_lam_')[1].split('_seed')[0]
        lam = float('inf') if lam_str == 'inf' else float(lam_str)
        df['lam'] = lam
        all_data.append(df)
        
    if not all_data: return
    df = pd.concat(all_data, ignore_index=True)
    df_layer0 = df[df['layer'] == 0]
    
    df_final = df_layer0[df_layer0['task'] >= 130]
    
    mean_stats = df_final.groupby(['lam', 'seed'])[['drift_rel', 'drift_rad', 'drift_tan', 'prev_tasks_acc']].mean().reset_index()
    final_stats = mean_stats.groupby('lam')[['drift_rel', 'drift_rad', 'drift_tan', 'prev_tasks_acc']].agg(['mean', 'std']).reset_index()
    
    finite = mean_stats[(mean_stats['lam'] != float('inf')) & (mean_stats['lam'] > 0)].copy()
    finite = finite.dropna()
    if not finite.empty:
        log_lam = np.log10(finite['lam'])
        log_drift = np.log10(finite['drift_rel'])
        res_alpha = stats.linregress(log_lam, log_drift)
        print(f"Drift vs Lambda Exponent (alpha-1): {res_alpha.slope:.4f} ± {res_alpha.stderr * stats.t.ppf(0.975, len(log_lam)-2):.4f}")
    
    mean_stats_no_na = mean_stats.dropna(subset=['drift_rel', 'prev_tasks_acc'])
    if not mean_stats_no_na.empty:
        res_ret = stats.linregress(mean_stats_no_na['drift_rel'], mean_stats_no_na['prev_tasks_acc'])
        print(f"Retention vs Drift: r={res_ret.rvalue:.4f}, p={res_ret.pvalue:.4e}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    f_final = final_stats[(final_stats['lam'] != float('inf')) & (final_stats['lam'] > 0)]
    
    if not f_final.empty:
        axes[0].errorbar(f_final['lam'], f_final['drift_rel']['mean'], yerr=f_final['drift_rel']['std'], fmt='-o')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Lambda')
        axes[0].set_ylabel('Relative Drift')
        axes[0].set_title('R3: Relative Drift vs Lambda')
        
        axes[1].scatter(mean_stats['drift_rel'], mean_stats['prev_tasks_acc'])
        axes[1].set_xlabel('Relative Drift')
        axes[1].set_ylabel('Retention (Prev Tasks Acc)')
        axes[1].set_title('R3: Retention tracks Drift')
        
        axes[2].errorbar(f_final['lam'], f_final['drift_rad']['mean'], yerr=f_final['drift_rad']['std'], fmt='-o', label='Radial Drift')
        axes[2].errorbar(f_final['lam'], f_final['drift_tan']['mean'], yerr=f_final['drift_tan']['std'], fmt='-s', label='Tangential Drift')
        axes[2].set_xscale('log')
        axes[2].set_xlabel('Lambda')
        axes[2].set_ylabel('Drift Component')
        axes[2].set_title('R3: Directional Suppression (Mechanism)')
        axes[2].legend()
        
    plt.tight_layout()
    plt.savefig('results/r3_drift_analysis.png')
    final_stats.to_csv('results/r3_drift_stats.csv', index=False)

if __name__ == '__main__':
    analyze_r3()
