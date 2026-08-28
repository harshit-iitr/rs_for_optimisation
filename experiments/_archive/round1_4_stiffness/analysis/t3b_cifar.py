import os
import pandas as pd

def analyze_t3b():
    base_dir = 'results'
    all_data = []
    
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R4_T3b_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        arm_name = run_id.split('_seed')[0].replace('R4_T3b_', '')
        df['arm_name'] = arm_name
        df['suite'] = 'R4_T3b'
        all_data.append(df)
        
    if not all_data:
        print("No data found for R4_T3b.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    df = df[df['layer'] == 0]
    
    # Filter for final task (task == 9)
    df_final = df[df['task'] == 9]
    
    # Aggregate over seeds
    stats = df_final.groupby(['arm_name', 'seed'])[['test_acc', 'prev_tasks_acc']].mean().reset_index()
    final_stats = stats.groupby('arm_name').agg(['mean', 'std', 'count']).reset_index()
    
    final_stats.columns = ['arm_name'] + [f'{col}_{stat}' for col, stat in final_stats.columns[1:]]
    
    print("=== T3b: Split CIFAR-100 Baselines ===")
    print(final_stats.sort_values('prev_tasks_acc_mean', ascending=False).to_string(index=False))
    final_stats.to_csv('results/t3b_cifar_stats.csv', index=False)

if __name__ == '__main__':
    analyze_t3b()
