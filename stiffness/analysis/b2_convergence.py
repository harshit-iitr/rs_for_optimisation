import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    eps = [1, 3, 10]
    seeds = [1, 2, 3]
    
    records = []
    
    for ep in eps:
        for seed in seeds:
            run_id = f"B2_ep_{ep}_seed{seed}"
            path = f"results/{run_id}/within_task.csv"
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            # steps are relative to the epoch, make them absolute
            df['abs_step'] = df['epoch'] * 235 + df['step'] # roughly 235 steps per epoch
            df['ep_setting'] = ep
            df['seed'] = seed
            records.append(df)
            
    if not records:
        print("No B2 records found.")
        return
        
    df = pd.concat(records)
    
    # Plot early tasks (e.g., task 0, 1, 2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, t_mask, title in zip(axes, [df['task'] < 3, df['task'] >= 47], ['First 3 Tasks', 'Last 3 Tasks']):
        sub = df[t_mask]
        for ep in eps:
            ep_sub = sub[sub['ep_setting'] == ep]
            if len(ep_sub) == 0:
                continue
            # aggregate by abs_step
            agg = ep_sub.groupby('abs_step')['test_acc'].agg(['mean', 'std']).reset_index()
            ax.plot(agg['abs_step'], agg['mean'], label=f'{ep} Epochs')
            ax.fill_between(agg['abs_step'], agg['mean'] - agg['std'], agg['mean'] + agg['std'], alpha=0.2)
            
        ax.set_title(title)
        ax.set_xlabel('Steps per task')
        ax.set_ylabel('Current Task Accuracy')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig('results/b2_convergence.png')
    print("Saved B2 plot to results/b2_convergence.png")
    
    # Determine which budget plateaus
    # Heuristically, print the max acc achieved at each budget
    for ep in eps:
        sub = df[df['ep_setting'] == ep]
        if len(sub) == 0: continue
        early_acc = sub[sub['task'] < 3]['test_acc'].max()
        late_acc = sub[sub['task'] >= 47]['test_acc'].max()
        print(f"Budget {ep} Epochs | Max Acc (Early): {early_acc:.4f} | Max Acc (Late): {late_acc:.4f}")

if __name__ == '__main__':
    main()
