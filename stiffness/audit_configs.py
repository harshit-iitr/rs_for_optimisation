import os
import json
import glob
import pandas as pd
import hashlib

def get_config_hash(config):
    # create a stable hash for config dict
    keys_to_hash = ['method', 'dataset', 'n_tasks', 'epochs', 'lr', 'batch_size', 'hidden_size', 'n_layers', 'optimizer', 'activation']
    hash_dict = {k: config.get(k) for k in keys_to_hash}
    s = json.dumps(hash_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]

def parse_run_id(run_id):
    if run_id.startswith('R1_') or run_id.startswith('R2_') or run_id.startswith('R3_') or run_id.startswith('R4_') or run_id.startswith('R5_') or run_id.startswith('R6_') or run_id.startswith('S5_') or run_id.startswith('A1') or run_id.startswith('B4'):
        parts = run_id.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return parts[0]
    # some fallbacks if naming isn't strict
    for p in ['C1', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'S5', 'A1', 'B4']:
        if p in run_id:
            return p
    return "UNKNOWN"

def main():
    base_dir = 'results'
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    records = []
    
    for d in dirs:
        conf_path = os.path.join(base_dir, d, 'config.json')
        if not os.path.exists(conf_path):
            continue
        try:
            with open(conf_path, 'r') as f:
                config = json.load(f)
        except Exception:
            continue
            
        suite = parse_run_id(d)
        if suite == 'UNKNOWN':
            # try to parse from the run_id prefix better
            if d.startswith('R'):
                parts = d.split('_')
                if len(parts) >= 2:
                    suite = f"{parts[0]}_{parts[1]}"
            else:
                suite = d.split('_')[0]
                
        # Handle specifically C1 which might be in R2_C1 or R3_C1
        if 'C1' in d:
            suite = 'C1'
        if 'B1' in d:
            suite = 'B1'
        if 'B4' in d:
            suite = 'B4'
            
        records.append({
            'suite': suite,
            'run_id': d,
            'epochs_per_task': config.get('epochs'),
            'lr': config.get('lr'),
            'n_tasks': config.get('n_tasks'),
            'width': config.get('hidden_size'),
            'depth': config.get('n_layers'),
            'optimizer': config.get('optimizer', 'sgd'),
            'activation': config.get('activation', 'relu'),
            'batch_size': config.get('batch_size'),
            'seed': config.get('seed'),
            'config_hash': get_config_hash(config)
        })
        
    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No configs found.")
        return
        
    # Group by everything except run_id and seed
    grouped = df.groupby(['suite', 'epochs_per_task', 'lr', 'n_tasks', 'width', 'depth', 'optimizer', 'activation', 'batch_size', 'config_hash']).agg(
        n_seeds=('seed', 'nunique')
    ).reset_index()
    
    grouped = grouped.sort_values(by=['suite'])
    
    md = "# Config Audit\n\n"
    md += "| Suite | Epochs/Task | LR | N Tasks | Width | Depth | Optimizer | Activation | Batch Size | N Seeds | Config Hash |\n"
    md += "|---|---|---|---|---|---|---|---|---|---|---|\n"
    for _, row in grouped.iterrows():
        md += f"| {row['suite']} | {row['epochs_per_task']} | {row['lr']} | {row['n_tasks']} | {row['width']} | {row['depth']} | {row['optimizer']} | {row['activation']} | {row['batch_size']} | {row['n_seeds']} | {row['config_hash']} |\n"
        
    with open('results/CONFIG_AUDIT.md', 'w') as f:
        f.write(md)
        
    print("Audit written to results/CONFIG_AUDIT.md")

if __name__ == '__main__':
    main()
