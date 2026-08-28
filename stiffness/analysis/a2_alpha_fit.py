import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
import scipy.stats as stats

def analyze_a2():
    base_dir = 'results'
    all_data = []
    
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R3_A2_lam_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        lam = float(run_id.split('_lam_')[1].split('_seed')[0])
        df['lam'] = lam
        all_data.append(df)
        
    if not all_data:
        print("No A2 data found.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    df = df[df['task'].isin([10, 25, 50])]
    
    results = []
    for layer in df['layer'].unique():
        for task in [10, 25, 50]:
            sub_df = df[(df['layer'] == layer) & (df['task'] == task)].copy()
            if sub_df.empty: continue
            
            sub_df = sub_df.dropna(subset=['g_rad_norm', 'radial_excess'])
            if sub_df.empty: continue
            
            mean_df = sub_df.groupby('lam')[['g_rad_norm', 'radial_excess']].mean().reset_index()
            mean_df = mean_df[mean_df['radial_excess'] > 0]
            if len(mean_df) < 3: continue
            
            log_lam = np.log10(mean_df['lam'])
            log_g_rad = np.log10(mean_df['g_rad_norm'])
            log_rad_ex = np.log10(mean_df['radial_excess'])
            
            res_alpha = linregress(log_lam, log_g_rad)
            alpha = res_alpha.slope
            ci = res_alpha.stderr * stats.t.ppf(0.975, len(log_lam)-2)
            
            res_beta = linregress(log_lam, log_rad_ex)
            obs_slope = res_beta.slope
            
            residual = obs_slope - (alpha - 1.0)
            
            results.append({
                'layer': layer,
                'task': task,
                'obs_slope': obs_slope,
                'alpha': alpha,
                'ci': ci,
                'predicted_slope': alpha - 1.0,
                'residual': residual
            })
            
    res_df = pd.DataFrame(results)
    print("=== A2: Alpha Exponent Fits ===")
    print(res_df.to_string(index=False))
    res_df.to_csv('results/a2_alpha_fit.csv', index=False)

if __name__ == '__main__':
    analyze_a2()
