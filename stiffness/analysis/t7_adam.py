import os
import pandas as pd

def analyze_t7():
    base_dir = 'results'
    all_data = []
    
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R3_R6_opt_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        
        parts = run_id.split('_')
        opt = run_id.split('opt_')[1].split('_lam')[0]
        lam = float(run_id.split('lam_')[1].split('_')[0])
        
        # Only keep adam and adamw
        if opt not in ['adam', 'adamw']: continue
            
        df['opt'] = opt
        df['lam'] = lam
        df['suite'] = 'R3_R6'
        all_data.append(df)
        
    if not all_data:
        print("No data found for Adam runs in R3_R6.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    df = df[df['layer'] == 0]
    
    # Filter for final 20 tasks
    df_final = df[df['task'] >= 130]
    
    # Aggregate over seeds
    stats = df_final.groupby(['opt', 'lam', 'seed'])[['test_acc', 'prev_tasks_acc']].mean().reset_index()
    final_stats = stats.groupby(['opt', 'lam']).agg(['mean', 'std', 'count']).reset_index()
    
    final_stats.columns = ['opt', 'lam'] + [f'{col}_{stat}' for col, stat in final_stats.columns[2:]]
    
    print("=== T7: Adam / AdamW Retention ===")
    print(final_stats.to_string(index=False))
    final_stats.to_csv('results/t7_adam_stats.csv', index=False)

if __name__ == '__main__':
    analyze_t7()
