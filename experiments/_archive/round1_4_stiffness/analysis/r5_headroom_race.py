import os
import pandas as pd
import numpy as np
from scipy import stats

def analyze_r5():
    base_dir = 'results'
    all_data = []
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R3_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        all_data.append(df)
        
    if not all_data: return
    df = pd.concat(all_data, ignore_index=True)
    
    df['normalized_gain'] = df['first_epoch_gain'] / (1.0 - (df['train_acc'] - df['first_epoch_gain']) + 1e-8)
    
    metrics = ['phi_rad_tilde', 'radial_excess', 'eff_rank', 'stable_rank', 'dead_frac', 'dormant_frac', 
               'weight_norm', 'grad_norm', 'update_norm', 'readiness', 'drift_cos_sim', 'drift_rel', 'drift_overlap']
               
    results = []
    
    grouped = df.groupby(['run_id', 'layer'])
    for name, group in grouped:
        for m in metrics:
            if m in group.columns and not group[m].isna().all():
                r, p = stats.spearmanr(group[m], group['normalized_gain'])
                if not np.isnan(r):
                    results.append({'run_id': name[0], 'layer': name[1], 'metric': m, 'r': r})
                    
    res_df = pd.DataFrame(results)
    if res_df.empty: return
    
    final_res = []
    for (layer, metric), group in res_df.groupby(['layer', 'metric']):
        r_vals = group['r'].dropna().values
        if len(r_vals) == 0: continue
        
        r_vals = np.clip(r_vals, -0.999, 0.999)
        z = np.arctanh(r_vals)
        z_mean = np.mean(z)
        pooled_r = np.tanh(z_mean)
        
        se = 1.0 / np.sqrt(len(group) - 3) if len(group) > 3 else 1.0
        z_stat = z_mean / se
        p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        final_res.append({'layer': layer, 'metric': metric, 'pooled_r': pooled_r, 'p_value': p})
        
    f_df = pd.DataFrame(final_res)
    f_df = f_df.sort_values(['layer', 'pooled_r'], ascending=[True, False])
    f_df.to_csv('results/r5_race.csv', index=False)
    print("R5 Headroom Race Results:")
    print(f_df)

if __name__ == '__main__':
    analyze_r5()
