import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_a3():
    base_dir = 'results'
    all_data = []
    
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('S1_lam_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        parts = run_id.split('_')
        lam_str = parts[2]
        lam = float('inf') if lam_str == 'inf' else float(lam_str)
        df['lam'] = lam
        df['suite'] = 'S1'
        all_data.append(df)
        
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('S2_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        arm_name = run_id.split('_seed')[0].replace('S2_', '')
        df['arm_name'] = arm_name
        df['suite'] = 'S2'
        all_data.append(df)
        
    if not all_data:
        print("No S1/S2 data found.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    df = df[df['layer'] == 0]
    
    df_final = df[df['task'] >= 130]
    
    df_s1 = df_final[df_final['suite'] == 'S1']
    s1_stats = df_s1.groupby(['lam', 'seed'])[['test_acc', 'prev_tasks_acc']].mean().reset_index()
    s1_final = s1_stats.groupby('lam')[['test_acc', 'prev_tasks_acc']].agg(['mean', 'std']).reset_index()
    s1_final.columns = ['lam', 'test_acc_mean', 'test_acc_std', 'prev_tasks_acc_mean', 'prev_tasks_acc_std']
    print("=== Round 1 S1 (Lambda Sweep) ===")
    print(s1_final.to_string(index=False))
    s1_final.to_csv('results/a3_s1_stats.csv', index=False)
    
    df_s2 = df_final[df_final['suite'] == 'S2']
    s2_stats = df_s2.groupby(['arm_name', 'seed'])[['test_acc', 'prev_tasks_acc']].mean().reset_index()
    s2_final = s2_stats.groupby('arm_name')[['test_acc', 'prev_tasks_acc']].agg(['mean', 'std']).reset_index()
    s2_final.columns = ['arm_name', 'test_acc_mean', 'test_acc_std', 'prev_tasks_acc_mean', 'prev_tasks_acc_std']
    print("\n=== Round 1 S2 (Baselines) ===")
    print(s2_final.sort_values('prev_tasks_acc_mean', ascending=False).to_string(index=False))
    s2_final.to_csv('results/a3_s2_stats.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    finite = s1_final[s1_final['lam'] != float('inf')]
    finite = finite[finite['lam'] > 0]
    plt.errorbar(finite['lam'], finite['prev_tasks_acc_mean'], yerr=finite['prev_tasks_acc_std'], fmt='-o')
    if float('inf') in s1_final['lam'].values:
        inf_val = s1_final[s1_final['lam'] == float('inf')]['prev_tasks_acc_mean'].values[0]
        plt.axhline(inf_val, color='r', linestyle='--', label='Hard Projection')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Retention')
    plt.title('A3 (Round 1 S1): Retention vs Lambda')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/a3_s1_retention.png')

if __name__ == '__main__':
    analyze_a3()
