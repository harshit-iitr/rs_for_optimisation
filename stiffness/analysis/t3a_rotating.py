import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_t3a():
    base_dir = 'results'
    all_data = []
    
    # Load S5_1 runs
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('S5_1_rot_lam_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        parts = run_id.split('_')
        lam_str = parts[4]
        lam = float('inf') if lam_str == 'inf' else float(lam_str)
        df['lam'] = lam
        df['suite'] = 'S5_1'
        all_data.append(df)
        
    # Load R4_T3a runs
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R4_T3a_rot_lam_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        parts = run_id.split('_')
        lam_str = parts[4]
        lam = float('inf') if lam_str == 'inf' else float(lam_str)
        df['lam'] = lam
        df['suite'] = 'R4_T3a'
        all_data.append(df)
        
    if not all_data:
        print("No data found for S5_1 or R4_T3a.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    df = df[df['layer'] == 0]
    
    # Filter for final 20 tasks (out of 100)
    df_final = df[df['task'] >= 80]
    
    # Aggregate over tasks and seeds
    stats = df_final.groupby(['lam', 'seed'])[['test_acc', 'prev_tasks_acc']].mean().reset_index()
    final_stats = stats.groupby('lam').agg(['mean', 'std', 'count']).reset_index()
    final_stats.columns = ['lam'] + [f'{col}_{stat}' for col, stat in final_stats.columns[1:]]
    
    print("=== T3a: Rotating MNIST Lambda Sweep ===")
    print(final_stats.to_string(index=False))
    final_stats.to_csv('results/t3a_rotating_bracket.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    finite = final_stats[final_stats['lam'] != float('inf')]
    finite = finite[finite['lam'] > 0.0]
    finite = finite.sort_values('lam')
    
    plt.errorbar(finite['lam'], finite['prev_tasks_acc_mean'], yerr=finite['prev_tasks_acc_std'], fmt='-o', color='blue', label='Retention')
    plt.errorbar(finite['lam'], finite['test_acc_mean'], yerr=finite['test_acc_std'], fmt='-s', color='orange', label='Current-Task Acc')
    
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('T3a: Rotating MNIST Retention vs Lambda')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/t3a_rotating_bracket.png')

if __name__ == '__main__':
    analyze_t3a()
