import os
import pandas as pd
from scipy.stats import ttest_rel

def analyze_t2c():
    base_dir = 'results'
    all_data = []
    
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R4_T2c_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        
        parts = run_id.split('_')
        if 'bp_iso' in run_id:
            method = 'bp_iso'
        elif '_bp_' in run_id:
            method = 'bp'
        elif '_rs_' in run_id:
            method = 'rs'
        else:
            continue
            
        df['method'] = method
        df['suite'] = 'R4_T2c'
        all_data.append(df)
        
    if not all_data:
        print("No data found for R4_T2c.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    df = df[df['layer'] == 0]
    
    # Filter for final 20 tasks
    df_final = df[df['task'] >= 130]
    
    # Aggregate over tasks
    stats = df_final.groupby(['method', 'seed'])[['test_acc', 'prev_tasks_acc']].mean().reset_index()
    
    # Paired t-tests
    methods = ['bp', 'bp_iso', 'rs']
    retention = {m: stats[stats['method'] == m].set_index('seed')['prev_tasks_acc'].sort_index() for m in methods}
    acc = {m: stats[stats['method'] == m].set_index('seed')['test_acc'].sort_index() for m in methods}
    
    final_stats = stats.groupby('method').agg(['mean', 'std', 'count']).reset_index()
    final_stats.columns = ['method'] + [f'{col}_{stat}' for col, stat in final_stats.columns[1:]]
    
    print("=== T2c: Canonical Isotropic Control ===")
    print(final_stats.to_string(index=False))
    
    # Comparisons
    print("\n--- Significance Tests (Retention) ---")
    
    # BP vs BP_ISO
    stat, p_val = ttest_rel(retention['bp'], retention['bp_iso'])
    wins = sum(retention['bp_iso'] > retention['bp'])
    print(f"BP_ISO vs BP: t={stat:.3f}, p={p_val:.4f}, n={len(retention['bp'])}, wins={wins}/{len(retention['bp'])}")
    
    # BP vs RS
    stat, p_val = ttest_rel(retention['bp'], retention['rs'])
    wins = sum(retention['rs'] > retention['bp'])
    print(f"RS vs BP: t={stat:.3f}, p={p_val:.4f}, n={len(retention['bp'])}, wins={wins}/{len(retention['bp'])}")
    
    # BP_ISO vs RS
    stat, p_val = ttest_rel(retention['bp_iso'], retention['rs'])
    wins = sum(retention['rs'] > retention['bp_iso'])
    print(f"RS vs BP_ISO: t={stat:.3f}, p={p_val:.4f}, n={len(retention['bp'])}, wins={wins}/{len(retention['bp'])}")
    
    # Calculate % magnitude recovered
    bp_ret = final_stats[final_stats['method'] == 'bp']['prev_tasks_acc_mean'].values[0]
    iso_ret = final_stats[final_stats['method'] == 'bp_iso']['prev_tasks_acc_mean'].values[0]
    rs_ret = final_stats[final_stats['method'] == 'rs']['prev_tasks_acc_mean'].values[0]
    
    total_gain = rs_ret - bp_ret
    iso_gain = iso_ret - bp_ret
    recovered = iso_gain / total_gain * 100 if total_gain != 0 else 0
    print(f"\nMagnitude matching recovers {recovered:.1f}% of the RS retention gain.")

if __name__ == '__main__':
    analyze_t2c()
