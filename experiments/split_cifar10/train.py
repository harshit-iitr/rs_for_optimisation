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

from shared.model import SimpleCNN
from shared.norm_penalty import norm_penalty
from shared.continual_backprop import ContinualBackpropResetter
from shared.diagnostics import (
    compute_effective_rank,
    compute_dead_neuron_fraction,
    compute_activation_norm_stats,
    compute_radial_energy,
    DiagnosticsTracker
)
from dataset import SplitCIFAR10Loader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_on_task(model, dataloader, task_idx, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    h_conv1_list = []
    h_conv2_list = []
    h_conv3_list = []
    h_fc_list = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, pre_acts = model(x, task_idx=task_idx)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * x.shape[0]
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.shape[0]
            
            if len(h_fc_list) * x.shape[0] < 1024:
                h_conv1_list.append(pre_acts[0].cpu())
                h_conv2_list.append(pre_acts[1].cpu())
                h_conv3_list.append(pre_acts[2].cpu())
                h_fc_list.append(pre_acts[3].cpu())
                
    avg_loss = total_loss / total
    accuracy = correct / total
    h_conv1 = torch.cat(h_conv1_list, dim=0) if h_conv1_list else torch.empty(0)
    h_conv2 = torch.cat(h_conv2_list, dim=0) if h_conv2_list else torch.empty(0)
    h_conv3 = torch.cat(h_conv3_list, dim=0) if h_conv3_list else torch.empty(0)
    h_fc = torch.cat(h_fc_list, dim=0) if h_fc_list else torch.empty(0)
    
    return avg_loss, accuracy, h_conv1, h_conv2, h_conv3, h_fc

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_base_dir = os.path.join(script_dir, "results")
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Load loaders once
    train_cifar = SplitCIFAR10Loader(root=data_dir, train=True, download=True)
    test_cifar = SplitCIFAR10Loader(root=data_dir, train=False, download=True)
    
    conditions = [
        {"name": "baseline", "lambda_norm_fc": 0.0, "lambda_norm_conv": 0.0, "weight_decay": 1e-3, "shrink_perturb": False, "cb": False},
        {"name": "strong_wd", "lambda_norm_fc": 0.0, "lambda_norm_conv": 0.0, "weight_decay": 1e-1, "shrink_perturb": False, "cb": False},
        {"name": "shrink_perturb", "lambda_norm_fc": 0.0, "lambda_norm_conv": 0.0, "weight_decay": 1e-3, "shrink_perturb": True, "cb": False},
        {"name": "norm_penalty_fc_only", "lambda_norm_fc": 0.05, "lambda_norm_conv": 0.0, "weight_decay": 1e-3, "shrink_perturb": False, "cb": False},
        {"name": "norm_penalty_all_layers", "lambda_norm_fc": 0.05, "lambda_norm_conv": 0.05, "weight_decay": 1e-3, "shrink_perturb": False, "cb": False},
        {"name": "continual_backprop", "lambda_norm_fc": 0.0, "lambda_norm_conv": 0.0, "weight_decay": 1e-3, "shrink_perturb": False, "cb": True},
        {"name": "norm_penalty_fc_plus_cb", "lambda_norm_fc": 0.05, "lambda_norm_conv": 0.0, "weight_decay": 1e-3, "shrink_perturb": False, "cb": True}
    ]
    
    seeds = [42, 43, 44]
    num_tasks = 5
    epochs_per_task = 40
    batch_size = 128
    criterion = nn.CrossEntropyLoss()
    
    # Pre-build task loaders to save time during loops
    train_loaders = []
    test_loaders = []
    for t in range(num_tasks):
        train_ds = train_cifar.get_task_dataset(t)
        test_ds = test_cifar.get_task_dataset(t)
        
        train_loaders.append(DataLoader(train_ds, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_ds, batch_size=batch_size, shuffle=False))
        
    for cond in conditions:
        cond_name = cond["name"]
        lambda_norm_fc = cond["lambda_norm_fc"]
        lambda_norm_conv = cond["lambda_norm_conv"]
        wd = cond["weight_decay"]
        do_shrink_perturb = cond["shrink_perturb"]
        do_cb = cond["cb"]
        
        print(f"\n--- Running condition: {cond_name} ---")
        
        for seed in seeds:
            print(f"Seed: {seed}")
            set_seed(seed)
            
            run_name = f"{cond_name}_seed_{seed}"
            tracker = DiagnosticsTracker(log_dir=results_base_dir, run_name=run_name)
            
            # SimpleCNN with 2 classes per head, 5 tasks, 256 hidden dim
            model = SimpleCNN(num_classes_per_head=2, num_tasks=num_tasks, d_hidden=256).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=wd)
            
            # Resetter for Continual Backprop
            cb_resetter = ContinualBackpropResetter(eta=0.01, threshold=0.01) if do_cb else None
            
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
                    
                    radial_energy_fc_sum = 0.0
                    batches_count = 0
                    
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        logits, pre_acts = model(x, task_idx=task_idx)
                        
                        h_conv1, h_conv2, h_conv3, h_fc = pre_acts
                        h_fc.retain_grad()
                        
                        loss_ce = criterion(logits, y)
                        
                        # Add L_norm penalty if active
                        loss_reg = 0.0
                        if lambda_norm_conv > 0:
                            loss_reg += norm_penalty(h_conv1) + norm_penalty(h_conv2) + norm_penalty(h_conv3)
                        if lambda_norm_fc > 0:
                            loss_reg += norm_penalty(h_fc)
                            
                        loss = loss_ce + loss_reg
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # Apply CB resetter after optimizer step
                        if do_cb:
                            activations = {
                                "conv1": torch.relu(h_conv1),
                                "conv2": torch.relu(h_conv2),
                                "conv3": torch.relu(h_conv3),
                                "fc_hidden": torch.relu(h_fc)
                            }
                            cb_resetter.update_and_reset(model, optimizer, activations)
                        
                        epoch_loss += loss_ce.item() * x.shape[0]
                        preds = torch.argmax(logits, dim=-1)
                        epoch_correct += (preds == y).sum().item()
                        epoch_total += y.shape[0]
                        
                        re_fc = compute_radial_energy(h_fc, h_fc.grad)
                        radial_energy_fc_sum += re_fc
                        batches_count += 1
                        
                    train_loss = epoch_loss / epoch_total
                    train_acc = epoch_correct / epoch_total
                        
                    # Evaluate on current task and Task 1
                    if epoch % 10 == 0 or epoch == 1 or epoch == epochs_per_task:
                        test_loss, test_acc, eval_c1, eval_c2, eval_c3, eval_fc = evaluate_on_task(model, test_loaders[task_idx], task_idx, criterion, device)
                        
                        # Evaluate on Task 1 for forgetting
                        _, task1_acc, _, _, _, _ = evaluate_on_task(model, test_loaders[0], 0, criterion, device)
                        
                        # Evaluate on all tasks seen so far
                        all_seen_accs = []
                        for t_prev in range(task_idx + 1):
                            _, t_acc, _, _, _, _ = evaluate_on_task(model, test_loaders[t_prev], t_prev, criterion, device)
                            all_seen_accs.append(t_acc)
                        avg_seen_acc = np.mean(all_seen_accs)
                        
                        rank_fc = compute_effective_rank(eval_fc)
                        dead_fc = compute_dead_neuron_fraction(eval_fc)
                        stats_fc = compute_activation_norm_stats(eval_fc)
                        
                        metrics = {
                            "task_index": task_idx,
                            "task_epoch": epoch,
                            "train_loss": train_loss,
                            "train_accuracy": train_acc,
                            "test_loss": test_loss,
                            "test_accuracy": test_acc, # Current task plasticity
                            "task1_accuracy": task1_acc, # Task 1 stability
                            "average_seen_accuracy": avg_seen_acc,
                            "effective_rank_fc": rank_fc,
                            "dead_neuron_fraction_fc": dead_fc,
                            "norm_stats_fc": stats_fc,
                            "radial_energy_fc": radial_energy_fc_sum / batches_count
                        }
                        tracker.log(global_epoch, metrics)
                        
                        if epoch == epochs_per_task:
                            print(f"    Task {task_idx+1} Done | Current Acc: {test_acc:.4f} | "
                                  f"Task 1 Acc: {task1_acc:.4f} | Avg Seen Acc: {avg_seen_acc:.4f} | "
                                  f"Dead FC: {dead_fc:.4f}")
                            
            tracker.save_checkpoint(global_epoch, model, optimizer, prefix="final_")
            
    print("\nSplit CIFAR-10 completed. Generating plots...")
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
                
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    colors_map = {
        "baseline": "blue",
        "strong_wd": "orange",
        "shrink_perturb": "brown",
        "norm_penalty_fc_only": "green",
        "norm_penalty_all_layers": "teal",
        "continual_backprop": "red",
        "norm_penalty_fc_plus_cb": "magenta"
    }
    
    for name, runs in data.items():
        if not runs:
            continue
        
        task_accs_per_seed = []
        task1_accs_per_seed = []
        dead_neurons_per_seed = []
        
        for run in runs:
            boundary_entries = [entry for entry in run if entry["task_epoch"] == 40]
            task_accs = [entry["test_accuracy"] for entry in boundary_entries]
            task1_accs = [entry["task1_accuracy"] for entry in boundary_entries]
            dead_neurons = [entry["dead_neuron_fraction_fc"] for entry in boundary_entries]
            
            task_accs_per_seed.append(task_accs)
            task1_accs_per_seed.append(task1_accs)
            dead_neurons_per_seed.append(dead_neurons)
            
        task_accs_per_seed = np.array(task_accs_per_seed) # (seeds, num_tasks)
        task1_accs_per_seed = np.array(task1_accs_per_seed) # (seeds, num_tasks)
        dead_neurons_per_seed = np.array(dead_neurons_per_seed) # (seeds, num_tasks)
        
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
        
        # Dead neurons plot
        mean_d = np.mean(dead_neurons_per_seed, axis=0)
        std_d = np.std(dead_neurons_per_seed, axis=0)
        axes[2].plot(tasks_x, mean_d, label=name, color=colors_map[name], marker='d', lw=2)
        axes[2].fill_between(tasks_x, mean_d - std_d, mean_d + std_d, color=colors_map[name], alpha=0.15)
        
    axes[0].set_title("Plasticity: Current Task Acc (Split CIFAR-10)")
    axes[0].set_xlabel("Task Index")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(np.arange(1, num_tasks + 1))
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    axes[1].set_title("Stability: Task 1 Acc (Split CIFAR-10)")
    axes[1].set_xlabel("Task Index (Trained up to)")
    axes[1].set_ylabel("Task 1 Accuracy")
    axes[1].set_xticks(np.arange(1, num_tasks + 1))
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    axes[2].set_title("Capacity: Dead Neuron Fraction (FC Layer)")
    axes[2].set_xlabel("Task Index")
    axes[2].set_ylabel("Dead Neuron Fraction")
    axes[2].set_xticks(np.arange(1, num_tasks + 1))
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "split_cifar10_comparison.png"), dpi=200)
    plt.close()
    print("Split CIFAR-10 plots saved successfully.")

if __name__ == "__main__":
    run_experiment()
