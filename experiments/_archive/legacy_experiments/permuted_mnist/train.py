import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import sys
import json

# Add parent directory to path so we can import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.model import SharedMLP
from shared.norm_penalty import norm_penalty
from shared.diagnostics import (
    compute_effective_rank,
    compute_dead_neuron_fraction,
    compute_activation_norm_stats,
    compute_radial_energy,
    DiagnosticsTracker
)
from dataset import PermutedMNISTLoader, get_permutations

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_on_task(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    h1_list = []
    h2_list = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, pre_acts = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.shape[0]
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.shape[0]
            
            if len(h1_list) * x.shape[0] < 2048:
                h1_list.append(pre_acts[0].cpu())
                h2_list.append(pre_acts[1].cpu())
                
    avg_loss = total_loss / total
    accuracy = correct / total
    h1_all = torch.cat(h1_list, dim=0) if h1_list else torch.empty(0)
    h2_all = torch.cat(h2_list, dim=0) if h2_list else torch.empty(0)
    
    return avg_loss, accuracy, h1_all, h2_all

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_base_dir = os.path.join(script_dir, "results")
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Load loaders once
    train_mnist = PermutedMNISTLoader(root=data_dir, train=True, download=True)
    test_mnist = PermutedMNISTLoader(root=data_dir, train=False, download=True)
    
    conditions = [
        {"name": "baseline", "lambda_norm": 0.0, "weight_decay": 1e-3, "l2_init": 0.0, "shrink_perturb": False},
        {"name": "strong_wd", "lambda_norm": 0.0, "weight_decay": 1e-1, "l2_init": 0.0, "shrink_perturb": False},
        {"name": "l2_init", "lambda_norm": 0.0, "weight_decay": 1e-3, "l2_init": 0.01, "shrink_perturb": False},
        {"name": "shrink_perturb", "lambda_norm": 0.0, "weight_decay": 1e-3, "l2_init": 0.0, "shrink_perturb": True},
        {"name": "norm_penalty", "lambda_norm": 0.05, "weight_decay": 1e-3, "l2_init": 0.0, "shrink_perturb": False}
    ]
    
    seeds = [42, 43, 44]
    num_tasks = 10
    epochs_per_task = 20
    batch_size = 256
    criterion = nn.CrossEntropyLoss()
    
    # Generate fixed permutations
    permutations = get_permutations(seed=1234, num_tasks=num_tasks)
    
    # Pre-build task loaders to save time during loops
    train_loaders = []
    test_loaders = []
    for t in range(num_tasks):
        train_ds = train_mnist.get_task_dataset(permutations[t])
        test_ds = test_mnist.get_task_dataset(permutations[t])
        
        train_loaders.append(DataLoader(train_ds, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_ds, batch_size=batch_size, shuffle=False))
        
    for cond in conditions:
        cond_name = cond["name"]
        lambda_norm = cond["lambda_norm"]
        wd = cond["weight_decay"]
        l2_init_coef = cond["l2_init"]
        do_shrink_perturb = cond["shrink_perturb"]
        
        print(f"\n--- Running condition: {cond_name} ---")
        
        for seed in seeds:
            print(f"Seed: {seed}")
            set_seed(seed)
            
            run_name = f"{cond_name}_seed_{seed}"
            tracker = DiagnosticsTracker(log_dir=results_base_dir, run_name=run_name)
            
            # Input dimension is 784, Output is 10 classes
            model = SharedMLP(d_in=784, d_out=10).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
            
            # Store initial weights for L2-init baseline
            init_weights = {name: param.clone().detach() for name, param in model.named_parameters()}
            
            global_epoch = 0
            
            for task_idx in range(num_tasks):
                print(f"  Training Task {task_idx+1}/{num_tasks}")
                
                # Apply Shrink and Perturb at task boundary (except for Task 1)
                if do_shrink_perturb and task_idx > 0:
                    with torch.no_grad():
                        for param in model.parameters():
                            param.data = 0.8 * param.data + 0.01 * torch.randn_like(param.data)
                
                train_loader = train_loaders[task_idx]
                
                for epoch in range(1, epochs_per_task + 1):
                    global_epoch += 1
                    model.train()
                    
                    epoch_loss = 0.0
                    epoch_correct = 0
                    epoch_total = 0
                    
                    radial_energy_h1_sum = 0.0
                    radial_energy_h2_sum = 0.0
                    batches_count = 0
                    
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        logits, pre_acts = model(x)
                        
                        h1, h2 = pre_acts[0], pre_acts[1]
                        h1.retain_grad()
                        h2.retain_grad()
                        
                        loss_ce = criterion(logits, y)
                        
                        # Add L_norm penalty if active
                        loss_reg = 0.0
                        if lambda_norm > 0:
                            loss_reg = norm_penalty(h1) + norm_penalty(h2)
                            
                        # Add L2 init penalty if active
                        loss_l2_init = 0.0
                        if l2_init_coef > 0:
                            for name, param in model.named_parameters():
                                loss_l2_init += torch.sum((param - init_weights[name]) ** 2)
                                
                        loss = loss_ce + lambda_norm * loss_reg + l2_init_coef * loss_l2_init
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss_ce.item() * x.shape[0]
                        preds = torch.argmax(logits, dim=-1)
                        epoch_correct += (preds == y).sum().item()
                        epoch_total += y.shape[0]
                        
                        re_h1 = compute_radial_energy(h1, h1.grad)
                        re_h2 = compute_radial_energy(h2, h2.grad)
                        radial_energy_h1_sum += re_h1
                        radial_energy_h2_sum += re_h2
                        batches_count += 1
                        
                    train_loss = epoch_loss / epoch_total
                    train_acc = epoch_correct / epoch_total
                        
                    # Evaluate on current task every 5 epochs
                    if epoch % 5 == 0 or epoch == 1 or epoch == epochs_per_task:
                        test_loss, test_acc, eval_h1, eval_h2 = evaluate_on_task(model, test_loaders[task_idx], criterion, device)
                        
                        # Evaluate on Task 1 for forgetting
                        _, task1_acc, _, _ = evaluate_on_task(model, test_loaders[0], criterion, device)
                        
                        # Evaluate on all tasks seen so far
                        all_seen_accs = []
                        for t_prev in range(task_idx + 1):
                            _, t_acc, _, _ = evaluate_on_task(model, test_loaders[t_prev], criterion, device)
                            all_seen_accs.append(t_acc)
                        avg_seen_acc = np.mean(all_seen_accs)
                        
                        rank_h1 = compute_effective_rank(eval_h1)
                        rank_h2 = compute_effective_rank(eval_h2)
                        
                        dead_h1 = compute_dead_neuron_fraction(eval_h1)
                        dead_h2 = compute_dead_neuron_fraction(eval_h2)
                        
                        stats_h1 = compute_activation_norm_stats(eval_h1)
                        stats_h2 = compute_activation_norm_stats(eval_h2)
                        
                        metrics = {
                            "task_index": task_idx,
                            "task_epoch": epoch,
                            "train_loss": train_loss,
                            "train_accuracy": train_acc,
                            "test_loss": test_loss,
                            "test_accuracy": test_acc, # Current task plasticity
                            "task1_accuracy": task1_acc, # Task 1 stability
                            "average_seen_accuracy": avg_seen_acc,
                            "effective_rank_h1": rank_h1,
                            "effective_rank_h2": rank_h2,
                            "dead_neuron_fraction_h1": dead_h1,
                            "dead_neuron_fraction_h2": dead_h2,
                            "norm_stats_h1": stats_h1,
                            "norm_stats_h2": stats_h2,
                            "radial_energy_h1": radial_energy_h1_sum / batches_count,
                            "radial_energy_h2": radial_energy_h2_sum / batches_count
                        }
                        tracker.log(global_epoch, metrics)
                        
                        if epoch == epochs_per_task:
                            print(f"    Task {task_idx+1} Done | Current Acc: {test_acc:.4f} | "
                                  f"Task 1 Acc: {task1_acc:.4f} | Avg Seen Acc: {avg_seen_acc:.4f} | "
                                  f"Dead H2: {dead_h2:.4f}")
                            
            tracker.save_checkpoint(global_epoch, model, optimizer, prefix="final_")
            
    print("\nPermuted MNIST completed. Generating plots...")
    generate_plots(results_base_dir, conditions, seeds, num_tasks)

def generate_plots(results_dir, conditions, seeds, num_tasks):
    data = {cond["name"]: [] for cond in conditions}
    for cond in conditions:
        name = cond["name"]
        for seed in seeds:
            json_path = os.path.join(results_dir, f"{name}_seed_{seed}_diagnostics.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    run_data = json.load(f)
                data[name].append(run_data)
                
    # We want a plot: x-axis: task number (1-10), y-axis: accuracy on that task at the end of its training
    # And another plot: x-axis: task number, y-axis: Task 1 accuracy after training on each subsequent task
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors_map = {
        "baseline": "blue",
        "strong_wd": "orange",
        "l2_init": "purple",
        "shrink_perturb": "brown",
        "norm_penalty": "green"
    }
    
    for name, runs in data.items():
        if not runs:
            continue
        
        # Collect peak current task accuracy at the end of each task (where task_epoch == 20)
        # runs is list of length (seeds) containing list of length (epochs logged)
        # We need to filter entries where task_epoch == 20
        task_accs_per_seed = []
        task1_accs_per_seed = []
        
        for run in runs:
            # Get list of entries at task boundary
            boundary_entries = [entry for entry in run if entry["task_epoch"] == 20]
            task_accs = [entry["test_accuracy"] for entry in boundary_entries]
            task1_accs = [entry["task1_accuracy"] for entry in boundary_entries]
            
            task_accs_per_seed.append(task_accs)
            task1_accs_per_seed.append(task1_accs)
            
        task_accs_per_seed = np.array(task_accs_per_seed) # (seeds, num_tasks)
        task1_accs_per_seed = np.array(task1_accs_per_seed) # (seeds, num_tasks)
        
        tasks_x = np.arange(1, num_tasks + 1)
        
        # Plasticity plot
        mean_p = np.mean(task_accs_per_seed, axis=0)
        std_p = np.std(task_accs_per_seed, axis=0)
        axes[0].plot(tasks_x, mean_p, label=name, color=colors_map[name], marker='o', lw=2)
        axes[0].fill_between(tasks_x, mean_p - std_p, mean_p + std_p, color=colors_map[name], alpha=0.15)
        
        # Forgetting plot
        mean_f = np.mean(task1_accs_per_seed, axis=0)
        std_f = np.std(task1_accs_per_seed, axis=0)
        axes[1].plot(tasks_x, mean_f, label=name, color=colors_map[name], marker='s', lw=2)
        axes[1].fill_between(tasks_x, mean_f - std_f, mean_f + std_f, color=colors_map[name], alpha=0.15)
        
    axes[0].set_title("Plasticity: Current Task Accuracy at End of Task Training")
    axes[0].set_xlabel("Task Index")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(np.arange(1, num_tasks + 1))
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    axes[1].set_title("Stability: Task 1 Accuracy over Sequential Learning")
    axes[1].set_xlabel("Task Index (Trained up to)")
    axes[1].set_ylabel("Task 1 Accuracy")
    axes[1].set_xticks(np.arange(1, num_tasks + 1))
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "permuted_mnist_comparison.png"), dpi=200)
    plt.close()
    print("Permuted MNIST plots saved successfully.")

if __name__ == "__main__":
    run_experiment()
