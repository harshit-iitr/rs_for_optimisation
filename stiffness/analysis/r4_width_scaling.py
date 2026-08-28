import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_r4():
    base_dir = 'results'
    all_data = []
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R3_R4_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        parts = run_id.split('_')
        width = int(parts[3])
        lam = float(parts[5])
        df['width'] = width
        df['lam'] = lam
        all_data.append(df)
        
    if not all_data: return
    df = pd.concat(all_data, ignore_index=True)
    df = df[df['layer'] == 0]
    
    df_final = df[df['task'] >= 130]
    stats = df_final.groupby(['width', 'lam', 'seed'])[['test_acc', 'prev_tasks_acc']].mean().reset_index()
    final_stats = stats.groupby(['width', 'lam'])[['test_acc', 'prev_tasks_acc']].mean().reset_index()
    
    widths = sorted(final_stats['width'].unique())
    bp_ret = []
    rs_ret = []
    for w in widths:
        bp = final_stats[(final_stats['width']==w) & (final_stats['lam']==0.0)]['prev_tasks_acc'].values
        rs = final_stats[(final_stats['width']==w) & (final_stats['lam']==0.03)]['prev_tasks_acc'].values
        bp_ret.append(bp[0] if len(bp) > 0 else float('nan'))
        rs_ret.append(rs[0] if len(rs) > 0 else float('nan'))
        
    delta = [r - b for r, b in zip(rs_ret, bp_ret)]
    
    plt.figure(figsize=(8, 6))
    plt.plot(widths, delta, marker='o', color='green', label='Retention Delta (RS - BP)')
    plt.xlabel('Width')
    plt.ylabel('Delta Retention')
    plt.title('R4: Width Scaling Control')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/r4_width_scaling.png')
    
    pd.DataFrame({'width': widths, 'bp_retention': bp_ret, 'rs_retention': rs_ret, 'delta_retention': delta}).to_csv('results/r4_stats.csv', index=False)

if __name__ == '__main__':
    analyze_r4()
