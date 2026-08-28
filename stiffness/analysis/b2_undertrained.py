import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_b2():
    base_dir = 'results'
    all_data = []
    
    # Read within_task.csv for B2 runs
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R2_B2_'): continue
        path = os.path.join(base_dir, run_id, 'within_task.csv')
        if not os.path.exists(path): continue
        
        ep = int(run_id.split('ep_')[1].split('_')[0])
        seed = int(run_id.split('seed')[-1])
        
        df = pd.read_csv(path)
        df['ep_budget'] = ep
        df['seed'] = seed
        all_data.append(df)
        
    if not all_data:
        print("No B2 data found.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    
    plt.figure(figsize=(12, 8))
    
    colors = {1: 'tab:blue', 3: 'tab:orange', 10: 'tab:green'}
    
    for ep in sorted(df['ep_budget'].unique()):
        df_ep = df[df['ep_budget'] == ep]
        if len(df_ep) == 0: continue
        
        c = colors.get(ep, 'k')
        
        # Early tasks
        df_early = df_ep[df_ep['task'] < 3]
        grouped_early = df_early.groupby('step')['test_acc'].agg(['mean', 'std']).reset_index()
        plt.plot(grouped_early['step'], grouped_early['mean'], label=f'Ep={ep} (Tasks 0-2)', linestyle='--', color=c)
        
        # Late tasks
        df_late = df_ep[df_ep['task'] >= 47]
        grouped_late = df_late.groupby('step')['test_acc'].agg(['mean', 'std']).reset_index()
        plt.plot(grouped_late['step'], grouped_late['mean'], label=f'Ep={ep} (Tasks 47-49)', linestyle='-', color=c)

    plt.xlabel('Steps within Task')
    plt.ylabel('Test Accuracy')
    plt.title('B2: Within-Task Convergence Plateau (lr=0.01)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/b2_undertrained.png')
    print("Saved plot to results/b2_undertrained.png")
    
if __name__ == '__main__':
    analyze_b2()
