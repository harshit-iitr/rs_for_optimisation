import os
import pandas as pd
import json

def extract():
    bp_data = []
    rs_data = []
    
    for s in range(1, 6):
        bp_path = f'results/R2_C1_A_bp_lr_0.1_seed{s}/metrics.parquet'
        rs_path = f'results/R2_C1_B_rs_lam_0.03_seed{s}/metrics.parquet'
        if os.path.exists(bp_path): bp_data.append(pd.read_parquet(bp_path))
        if os.path.exists(rs_path): rs_data.append(pd.read_parquet(rs_path))
        
    bp_df = pd.concat(bp_data)
    rs_df = pd.concat(rs_data)
    
    bp_mean = bp_df.groupby(['task', 'layer'])['grad_norm'].mean().reset_index()
    rs_mean = rs_df.groupby(['task', 'layer'])['grad_norm'].mean().reset_index()
    
    merged = pd.merge(bp_mean, rs_mean, on=['task', 'layer'], suffixes=('_bp', '_rs'))
    merged['s'] = merged['grad_norm_rs'] / merged['grad_norm_bp']
    
    out = {}
    for _, row in merged.iterrows():
        t = int(row['task'])
        l = int(row['layer'])
        s = float(row['s'])
        if t not in out: out[t] = {}
        out[t][l] = s
        
    with open('results/iso_shrinkage.json', 'w') as f:
        json.dump(out, f, indent=2)
        
    print("Extracted shrinkage factors to results/iso_shrinkage.json")

if __name__ == '__main__':
    extract()
