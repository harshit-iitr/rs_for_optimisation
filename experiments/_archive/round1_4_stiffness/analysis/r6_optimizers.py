import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_r6():
    base_dir = 'results'
    all_data = []
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R3_R6_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        parts = run_id.split('_')
        opt = parts[3]
        if parts[4] == 'momentum':
            opt = 'sgd_momentum'
            lam = float(parts[6])
        else:
            lam = float(parts[5])
        df['opt'] = opt
        df['lam'] = lam
        all_data.append(df)
        
    if not all_data: return
    df = pd.concat(all_data, ignore_index=True)
    df = df[df['layer'] == 0]
    
    df_final = df[df['task'] >= 130]
    stats = df_final.groupby(['opt', 'lam', 'seed'])[['test_acc', 'prev_tasks_acc', 'update_norm']].mean().reset_index()
    final_stats = stats.groupby(['opt', 'lam'])[['test_acc', 'prev_tasks_acc', 'update_norm']].agg(['mean', 'std']).reset_index()
    
    final_stats.columns = ['opt', 'lam', 'test_acc_mean', 'test_acc_std', 'prev_tasks_acc_mean', 'prev_tasks_acc_std', 'update_norm_mean', 'update_norm_std']
    
    plt.figure(figsize=(10, 8))
    
    for opt in final_stats['opt'].unique():
        opt_data = final_stats[final_stats['opt'] == opt].sort_values('lam')
        plt.errorbar(opt_data['test_acc_mean'], opt_data['prev_tasks_acc_mean'], 
                     xerr=opt_data['test_acc_std'], yerr=opt_data['prev_tasks_acc_std'], 
                     fmt='-o', label=opt)
                     
        for _, row in opt_data.iterrows():
            marker = 'BP' if row['lam'] == 0.0 else f"RS({row['lam']})"
            plt.annotate(marker, (row['test_acc_mean'], row['prev_tasks_acc_mean']), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
                         
    plt.xlabel('Current Task Accuracy')
    plt.ylabel('Previous Tasks Retention')
    plt.title('R6: Optimizer Retention Collapse')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/r6_optimizers.png')
    
    final_stats.to_csv('results/r6_stats.csv', index=False)

if __name__ == '__main__':
    analyze_r6()
