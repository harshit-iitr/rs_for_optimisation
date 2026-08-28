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
from dataset import MNISTAdditionDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, criterion, device):
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
    
    # 3 Conditions
    conditions = [
        {"name": "baseline", "lambda_norm": 0.0, "weight_decay": 1e-3},
        {"name": "strong_wd", "lambda_norm": 0.0, "weight_decay": 1e-1},
        {"name": "norm_penalty", "lambda_norm": 0.05, "weight_decay": 1e-3}
    ]
    
    seeds = [42, 43, 44]
    epochs = 500
    batch_size = 256
    
    # Construct datasets
    train_dataset = MNISTAdditionDataset(root=data_dir, train=True, download=True, f_train=0.3, num_pairs=10000, seed=42)
    test_dataset = MNISTAdditionDataset(root=data_dir, train=False, download=True, f_train=0.3, num_pairs=5000, seed=42)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    criterion = nn.CrossEntropyLoss()
    
    for cond in conditions:
        cond_name = cond["name"]
        lambda_norm = cond["lambda_norm"]
        wd = cond["weight_decay"]
        
        print(f"\n--- Running condition: {cond_name} ---")
        
        for seed in seeds:
            print(f"Seed: {seed}")
            set_seed(seed)
            
            run_name = f"{cond_name}_seed_{seed}"
            tracker = DiagnosticsTracker(log_dir=results_base_dir, run_name=run_name)
            
            # Input dimension is 1568 (28x56 concatenated images), Output is 10 classes (modulo 10 sum)
            model = SharedMLP(d_in=1568, d_out=10).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
            
            for epoch in range(1, epochs + 1):
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
                    loss_reg = 0.0
                    if lambda_norm > 0:
                        loss_reg = norm_penalty(h1) + norm_penalty(h2)
                        
                    loss = loss_ce + lambda_norm * loss_reg
                    
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
                
                # Log metrics every 10 epochs
                if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                    test_loss, test_acc, eval_h1, eval_h2 = evaluate(model, test_loader, criterion, device)
                    
                    rank_h1 = compute_effective_rank(eval_h1)
                    rank_h2 = compute_effective_rank(eval_h2)
                    
                    dead_h1 = compute_dead_neuron_fraction(eval_h1)
                    dead_h2 = compute_dead_neuron_fraction(eval_h2)
                    
                    stats_h1 = compute_activation_norm_stats(eval_h1)
                    stats_h2 = compute_activation_norm_stats(eval_h2)
                    
                    metrics = {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "test_loss": test_loss,
                        "test_accuracy": test_acc,
                        "effective_rank_h1": rank_h1,
                        "effective_rank_h2": rank_h2,
                        "dead_neuron_fraction_h1": dead_h1,
                        "dead_neuron_fraction_h2": dead_h2,
                        "norm_stats_h1": stats_h1,
                        "norm_stats_h2": stats_h2,
                        "radial_energy_h1": radial_energy_h1_sum / batches_count,
                        "radial_energy_h2": radial_energy_h2_sum / batches_count
                    }
                    tracker.log(epoch, metrics)
                    
                    if epoch % 50 == 0 or epoch == 1 or epoch == epochs:
                        print(f"Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
                              f"Rank H2: {rank_h2:.2f} | Dead H2: {dead_h2:.4f} | Norm H2 Mean: {stats_h2['mean']:.2f}")
                        
            tracker.save_checkpoint(epochs, model, optimizer, prefix="final_")
            
    print("\nMNIST Addition completed. Generating plots...")
    generate_plots(results_base_dir, conditions, seeds, epochs)

def generate_plots(results_dir, conditions, seeds, max_epochs):
    data = {cond["name"]: [] for cond in conditions}
    for cond in conditions:
        name = cond["name"]
        for seed in seeds:
            json_path = os.path.join(results_dir, f"{name}_seed_{seed}_diagnostics.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    run_data = json.load(f)
                data[name].append(run_data)
                
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors_map = {
        "baseline": "blue",
        "strong_wd": "orange",
        "norm_penalty": "green"
    }
    
    for name, runs in data.items():
        if not runs:
            continue
        
        epochs = [entry["epoch"] for entry in runs[0]]
        
        # Test Accuracy
        test_accs = np.array([[entry["test_accuracy"] for entry in run] for run in runs])
        mean_acc = np.mean(test_accs, axis=0)
        std_acc = np.std(test_accs, axis=0)
        axes[0].plot(epochs, mean_acc, label=name, color=colors_map[name], lw=2)
        axes[0].fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color=colors_map[name], alpha=0.15)
        
        # Train Accuracy
        train_accs = np.array([[entry["train_accuracy"] for entry in run] for run in runs])
        mean_train = np.mean(train_accs, axis=0)
        std_train = np.std(train_accs, axis=0)
        axes[1].plot(epochs, mean_train, label=f"{name} (Train)", color=colors_map[name], linestyle="--", lw=1.5)
        axes[1].plot(epochs, mean_acc, label=f"{name} (Test)", color=colors_map[name], lw=2)
        
    axes[0].set_title("MNIST Addition: Test Accuracy (Generalization on Unseen Sums)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    axes[1].set_title("MNIST Addition: Train vs Test Accuracy (Grokking Curves)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "digit_addition_comparison.png"), dpi=200)
    plt.close()
    print("MNIST Addition plots saved successfully.")

if __name__ == "__main__":
    run_experiment()
