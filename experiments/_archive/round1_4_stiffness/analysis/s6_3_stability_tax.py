import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_stability_tax():
    base_dir = 'results'
    if not os.path.exists(base_dir):
        print("No results found.")
        return

    # We want BP, RS(1.0), L2-init from S6.1 (which was run with the updated train.py logging prev_tasks_acc)
    target_methods = ['bp', 'rs', 'l2_init']
    
    all_data = []
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('S6_1_lr_0.01_'):
            continue
            
        parquet_path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            # Filter for specific methods
            method = df['method'].iloc[0]
            if method == 'rs' and df['lambda_rs'].iloc[0] != 1.0:
                continue
                
            if method in target_methods:
                all_data.append(df)
                
    if not all_data:
        print("No relevant metrics data found for S6.3.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    
    # We want to plot task (x) vs prev_tasks_acc (y)
    # Average across layers first
    df_net = df.groupby(['method', 'seed', 'task']).mean(numeric_only=True).reset_index()
    
    # Average across seeds
    df_agg = df_net.groupby(['method', 'task']).agg({'prev_tasks_acc': ['mean', 'std']}).reset_index()
    
    plt.figure(figsize=(10, 6))
    for method in target_methods:
        method_data = df_agg[df_agg['method'] == method]
        if len(method_data) == 0:
            continue
            
        x = method_data['task']
        y = method_data['prev_tasks_acc']['mean']
        y_err = method_data['prev_tasks_acc']['std']
        
        plt.plot(x, y, label=method)
        plt.fill_between(x, y - y_err, y + y_err, alpha=0.2)
        
    plt.xlabel("Task")
    plt.ylabel("Accuracy on all previous tasks (Stability)")
    plt.title("S6.3: Stability Tax (Forgetting Frontier)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(base_dir, 's6_3_stability_tax.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved stability tax plot to {out_path}")

if __name__ == '__main__':
    plot_stability_tax()
