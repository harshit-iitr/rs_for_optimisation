import os
import pandas as pd
import scipy.stats as stats

def run_analysis():
    all_data = []
    base_dir = 'results'
    if not os.path.exists(base_dir):
        print("No results found.")
        return

    # Gather data from S1 and S2
    for run_id in os.listdir(base_dir):
        if (run_id.startswith('S1_') or run_id.startswith('S2_')) and os.path.isdir(os.path.join(base_dir, run_id)):
            parquet_path = os.path.join(base_dir, run_id, 'metrics.parquet')
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
                all_data.append(df)
                
    if not all_data:
        print("No metrics data found.")
        return

    df = pd.concat(all_data, ignore_index=True)
    
    # We want to correlate metric at task t with first_epoch_gain at task t+1
    # So we shift first_epoch_gain backwards by 1 task for each run_id and layer
    
    metrics = [
        'phi_rad_tilde', 'eff_rank', 'stable_rank', 
        'dead_frac', 'dormant_frac', 'weight_norm', 'readiness'
    ]
    
    # We average over layers first to get network-wide metrics for each task
    df_net = df.groupby(['run_id', 'task', 'method', 'lambda_rs']).mean(numeric_only=True).reset_index()
    
    # Shift target
    df_net['next_gain'] = df_net.groupby('run_id')['first_epoch_gain'].shift(-1)
    df_net = df_net.dropna(subset=['next_gain'])
    
    def compute_corrs(subset_df, name):
        print(f"\n--- Correlations: {name} (n={len(subset_df)}) ---")
        for m in metrics:
            if m not in subset_df.columns:
                continue
            x = subset_df[m]
            y = subset_df['next_gain']
            spearman_r, spearman_p = stats.spearmanr(x, y)
            kendall_tau, kendall_p = stats.kendalltau(x, y)
            print(f"{m:15s} | Spearman: {spearman_r:6.3f} (p={spearman_p:.1e}) | Kendall: {kendall_tau:6.3f} (p={kendall_p:.1e})")

    # 1. Within lambda=0 arm alone (method == 'bp' or lambda_rs == 0.0)
    df_lam0 = df_net[(df_net['method'] == 'bp') | ((df_net['method'] == 'rs') & (df_net['lambda_rs'] == 0.0))]
    compute_corrs(df_lam0, "lambda=0 only")
    
    # 2. Pooled across all arms
    compute_corrs(df_net, "Pooled across all S1/S2 arms")
    
    print("\nDiagnostic race complete.")

if __name__ == '__main__':
    run_analysis()
