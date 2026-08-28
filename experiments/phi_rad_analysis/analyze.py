import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def analyze_phi_rad():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "permuted_mnist", "results")
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    seeds = [42, 43, 44]
    conditions = ["baseline", "strong_wd", "l2_init", "shrink_perturb", "norm_penalty"]
    
    # We want to collect pairs of (radial_energy of task t, peak_accuracy of task t+1)
    # let's analyze layer h2 as it's the penultimate layer
    
    print("Starting Φ_rad correlation analysis...")
    
    # Let's collect data per condition to analyze correlation within each condition
    all_data = {}
    
    for cond in conditions:
        phi_rad_values = []
        next_task_accs = []
        
        for seed in seeds:
            json_path = os.path.join(results_dir, f"{cond}_seed_{seed}_diagnostics.json")
            if not os.path.exists(json_path):
                print(f"Warning: {json_path} does not exist. Skipping.")
                continue
                
            with open(json_path, "r") as f:
                history = json.load(f)
                
            # Filter entries at the end of each task (task_epoch == 20)
            boundary_entries = [entry for entry in history if entry["task_epoch"] == 20]
            
            # For each task t (from 0 to 8), get radial energy, and for task t+1, get test_accuracy
            for t in range(len(boundary_entries) - 1):
                # Radial energy during task t (logged at epoch 20 of task t, which averages over task t)
                phi = boundary_entries[t]["radial_energy_h2"]
                # Accuracy on next task t+1 at the end of its training
                acc = boundary_entries[t+1]["test_accuracy"]
                
                phi_rad_values.append(phi)
                next_task_accs.append(acc)
                
        if len(phi_rad_values) > 1:
            r_val, p_val = pearsonr(phi_rad_values, next_task_accs)
            print(f"Condition: {cond:15s} | Pearson r: {r_val:6.3f} | p-value: {p_val:.3e} | N: {len(phi_rad_values)}")
            all_data[cond] = {
                "phi": phi_rad_values,
                "acc": next_task_accs,
                "r": r_val,
                "p": p_val
            }
            
    # Generate scatter plot comparison
    fig, axes = plt.subplots(1, len(all_data), figsize=(20, 4), sharey=True)
    if len(all_data) == 1:
        axes = [axes]
        
    for i, (cond, data) in enumerate(all_data.items()):
        ax = axes[i]
        ax.scatter(data["phi"], data["acc"], alpha=0.7, edgecolors='none', s=40, color='crimson')
        # Add regression line
        m, b = np.polyfit(data["phi"], data["acc"], 1)
        x_vals = np.linspace(min(data["phi"]), max(data["phi"]), 100)
        ax.plot(x_vals, m*x_vals + b, color='black', linestyle='--', alpha=0.8)
        
        ax.set_title(f"{cond}\n(r = {data['r']:.2f}, p = {data['p']:.2e})")
        ax.set_xlabel("Φ_rad (h2) at Task t")
        if i == 0:
            ax.set_ylabel("Accuracy at Task t+1")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phi_rad_vs_next_acc.png"), dpi=200)
    plt.close()
    
    # Generate overlay plot for baseline vs norm_penalty
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Let's plot baseline averages per task index
    baseline_phi = []
    baseline_acc = []
    np_phi = []
    np_acc = []
    
    # Average across seeds for each task index (0 to 9)
    for t_idx in range(10):
        # Baseline
        t_phi_b, t_acc_b = [], []
        t_phi_n, t_acc_n = [], []
        for seed in seeds:
            # Baseline
            json_b = os.path.join(results_dir, f"baseline_seed_{seed}_diagnostics.json")
            if os.path.exists(json_b):
                with open(json_b, "r") as f:
                    history = json.load(f)
                b_entries = [entry for entry in history if entry["task_epoch"] == 20]
                t_phi_b.append(b_entries[t_idx]["radial_energy_h2"])
                t_acc_b.append(b_entries[t_idx]["test_accuracy"])
            # Norm penalty
            json_n = os.path.join(results_dir, f"norm_penalty_seed_{seed}_diagnostics.json")
            if os.path.exists(json_n):
                with open(json_n, "r") as f:
                    history = json.load(f)
                n_entries = [entry for entry in history if entry["task_epoch"] == 20]
                t_phi_n.append(n_entries[t_idx]["radial_energy_h2"])
                t_acc_n.append(n_entries[t_idx]["test_accuracy"])
                
        baseline_phi.append(np.mean(t_phi_b))
        baseline_acc.append(np.mean(t_acc_b))
        np_phi.append(np.mean(t_phi_n))
        np_acc.append(np.mean(t_acc_n))
        
    tasks = np.arange(1, 11)
    
    color = 'tab:red'
    ax1.set_xlabel('Task Index')
    ax1.set_ylabel('Φ_rad (h2)', color=color)
    line1 = ax1.plot(tasks, baseline_phi, color=color, linestyle='-', marker='o', label='Baseline Φ_rad')
    line2 = ax1.plot(tasks, np_phi, color='tomato', linestyle='--', marker='x', label='Norm Penalty Φ_rad')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Next Task Accuracy', color=color)  
    line3 = ax2.plot(tasks[:-1], baseline_acc[1:], color=color, linestyle='-', marker='s', label='Baseline Acc (t+1)')
    line4 = ax2.plot(tasks[:-1], np_acc[1:], color='skyblue', linestyle='--', marker='^', label='Norm Penalty Acc (t+1)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left')
    
    plt.title("Φ_rad (Task t) vs. Accuracy (Task t+1) Overlay")
    fig.tight_layout()  
    plt.savefig(os.path.join(plots_dir, "phi_rad_acc_overlay.png"), dpi=200)
    plt.close()
    
    print("Φ_rad diagnostic plots saved in experiments/phi_rad_analysis/plots/")

if __name__ == "__main__":
    analyze_phi_rad()
