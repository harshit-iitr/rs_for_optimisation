import os
import json
import pandas as pd
import numpy as np

def compute_iso_shrinkage():
    base_dir = 'results'
    
    # We will use the runs from T1 (lambda=0 for BP, lambda=0.01 for RS)
    bp_runs = []
    rs_runs = []
    
    for seed in [1, 2, 3, 4, 5]:
        bp_path = os.path.join(base_dir, f'R4_T1_lam_0.0_seed{seed}', 'metrics.parquet')
        rs_path = os.path.join(base_dir, f'R4_T1_lam_0.01_seed{seed}', 'metrics.parquet')
        
        if os.path.exists(bp_path):
            bp_runs.append(pd.read_parquet(bp_path))
        if os.path.exists(rs_path):
            rs_runs.append(pd.read_parquet(rs_path))
            
    if not bp_runs or not rs_runs:
        print("Waiting for BP and RS runs from R4_T1 to compute shrinkage.")
        return
        
    df_bp = pd.concat(bp_runs)
    df_rs = pd.concat(rs_runs)
    
    # We want the ratio of weight_norm at the end of each task
    # shrinkage(task, layer) = RS_weight_norm / BP_weight_norm
    
    mean_bp = df_bp.groupby(['task', 'layer'])['weight_norm'].mean().reset_index()
    mean_rs = df_rs.groupby(['task', 'layer'])['weight_norm'].mean().reset_index()
    
    shrinkage = {}
    for task in sorted(mean_bp['task'].unique()):
        shrinkage[str(int(task))] = {}
        for layer in sorted(mean_bp['layer'].unique()):
            bp_val = mean_bp[(mean_bp['task'] == task) & (mean_bp['layer'] == layer)]['weight_norm'].values[0]
            rs_val = mean_rs[(mean_rs['task'] == task) & (mean_rs['layer'] == layer)]['weight_norm'].values[0]
            shrinkage[str(int(task))][str(int(layer))] = float(rs_val / bp_val)
            
    with open('results/iso_shrinkage.json', 'w') as f:
        json.dump(shrinkage, f, indent=2)
    print("Generated results/iso_shrinkage.json for T2c")

if __name__ == '__main__':
    compute_iso_shrinkage()
