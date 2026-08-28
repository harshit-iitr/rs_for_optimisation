import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms.functional as TF
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
from shared.continual_backprop import ContinualBackpropResetter
from shared.diagnostics import (
    compute_effective_rank,
    compute_dead_neuron_fraction,
    compute_activation_norm_stats,
    compute_radial_energy,
    DiagnosticsTracker
)
from dataset import RotatingMNISTLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_rotated_loader(dataset_loader, angle, batch_size, shuffle=True):
    # Rotate images on the fly and construct a DataLoader
    # dataset_loader.images is (N, 1, 28, 28)
    if angle == 0:
        rotated_imgs = dataset_loader.images.view(-1, 784)
    else:
        rotated_imgs = TF.rotate(dataset_loader.images, angle)
        rotated_imgs = rotated_imgs.view(-1, 784)
        
    ds = TensorDataset(rotated_imgs, dataset_loader.labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def evaluate_on_rotation(model, dataset_loader, angle, criterion, device):
    model.eval()
    dataloader = get_rotated_loader(dataset_loader, angle, batch_size=1024, shuffle=False)
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
    train_mnist = RotatingMNISTLoader(root=data_dir, train=True, download=True)
    test_mnist = RotatingMNISTLoader(root=data_dir, train=False, download=True)
    
    conditions = [
        {"name": "baseline", "lambda_norm": 0.0, "weight_decay": 1e-3, "cb": False},
        {"name": "strong_wd", "lambda_norm": 0.0, "weight_decay": 1e-1, "cb": False},
        {"name": "norm_penalty", "lambda_norm": 0.05, "weight_decay": 1e-3, "cb": False},
        {"name": "continual_backprop", "lambda_norm": 0.0, "weight_decay": 1e-3, "cb": True},
        {"name": "norm_penalty_plus_cb", "lambda_norm": 0.05, "weight_decay": 1e-3, "cb": True}
    ]
    
    seeds = [42, 43, 44]
    total_epochs = 100
    batch_size = 256
    criterion = nn.CrossEntropyLoss()
    
    for cond in conditions:
        cond_name = cond["name"]
        lambda_norm = cond["lambda_norm"]
        wd = cond["weight_decay"]
        do_cb = cond["cb"]
        
        print(f"\n--- Running condition: {cond_name} ---")
        
        for seed in seeds:
            print(f"Seed: {seed}")
            set_seed(seed)
            
            run_name = f"{cond_name}_seed_{seed}"
            tracker = DiagnosticsTracker(log_dir=results_base_dir, run_name=run_name)
            
            # SharedMLP with 784 in, 10 out, 256 hidden
            model = SharedMLP(d_in=784, d_out=10, d_hidden=256).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
            
            cb_resetter = ContinualBackpropResetter(eta=0.01, threshold=0.01) if do_cb else None
            
            for epoch in range(1, total_epochs + 1):
                # Continuous rotation: angle increases from 0 to 180 degrees
                angle = 180.0 * (epoch - 1) / total_epochs
                
                # Get rotated loader for this epoch
                train_loader = get_rotated_loader(train_mnist, angle, batch_size, shuffle=True)
                
                model.train()
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                
                radial_energy_h2_sum = 0.0
                batches_count = 0
                
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    logits, pre_acts = model(x)
                    
                    h1, h2 = pre_acts
                    h2.retain_grad()
                    
                    loss_ce = criterion(logits, y)
                    loss_reg = 0.0
                    if lambda_norm > 0:
                        loss_reg = norm_penalty(h1) + norm_penalty(h2)
                        
                    loss = loss_ce + lambda_norm * loss_reg
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if do_cb:
                        activations = {
                            "h1": torch.relu(h1),
                            "h2": torch.relu(h2)
                        }
                        cb_resetter.update_and_reset(model, optimizer, activations)
                        
                    epoch_loss += loss_ce.item() * x.shape[0]
                    preds = torch.argmax(logits, dim=-1)
                    epoch_correct += (preds == y).sum().item()
                    epoch_total += y.shape[0]
                    
                    re_h2 = compute_radial_energy(h2, h2.grad)
                    radial_energy_h2_sum += re_h2
                    batches_count += 1
                    
                train_loss = epoch_loss / epoch_total
                train_acc = epoch_correct / epoch_total
                
                # Evaluate on the current angle's test set
                if epoch % 5 == 0 or epoch == 1 or epoch == total_epochs:
                    test_loss, test_acc, eval_h1, eval_h2 = evaluate_on_rotation(model, test_mnist, angle, criterion, device)
                    
                    rank_h2 = compute_effective_rank(eval_h2)
                    dead_h2 = compute_dead_neuron_fraction(eval_h2)
                    stats_h2 = compute_activation_norm_stats(eval_h2)
                    
                    metrics = {
                        "epoch": epoch,
                        "angle": angle,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "test_loss": test_loss,
                        "test_accuracy": test_acc,
                        "effective_rank_h2": rank_h2,
                        "dead_neuron_fraction_h2": dead_h2,
                        "norm_stats_h2": stats_h2,
                        "radial_energy_h2": radial_energy_h2_sum / batches_count
                    }
                    tracker.log(epoch, metrics)
                    
                    if epoch % 20 == 0 or epoch == total_epochs:
                        print(f"    Epoch {epoch:3d} (Angle {angle:5.1f}°) | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Dead H2: {dead_h2:.4f}")
                        
            tracker.save_checkpoint(total_epochs, model, optimizer, prefix="final_")
            
    print("\nRotating MNIST completed. Generating plots...")
    generate_plots(results_base_dir, conditions, seeds, total_epochs)

def generate_plots(results_dir, conditions, seeds, total_epochs):
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
        "norm_penalty": "green",
        "continual_backprop": "red",
        "norm_penalty_plus_cb": "magenta"
    }
    
    for name, runs in data.items():
        if not runs:
            continue
        
        accs_per_seed = []
        dead_neurons_per_seed = []
        ranks_per_seed = []
        epochs_logged = []
        
        for run in runs:
            accs = [entry["test_accuracy"] for entry in run]
            dead_neurons = [entry["dead_neuron_fraction_h2"] for entry in run]
            ranks = [entry["effective_rank_h2"] for entry in run]
            epochs_logged = [entry["epoch"] for entry in run]
            
            accs_per_seed.append(accs)
            dead_neurons_per_seed.append(dead_neurons)
            ranks_per_seed.append(ranks)
            
        accs_per_seed = np.array(accs_per_seed)
        dead_neurons_per_seed = np.array(dead_neurons_per_seed)
        ranks_per_seed = np.array(ranks_per_seed)
        
        x = np.array(epochs_logged)
        
        # Test Accuracy
        mean_acc = np.mean(accs_per_seed, axis=0)
        std_acc = np.std(accs_per_seed, axis=0)
        axes[0].plot(x, mean_acc, label=name, color=colors_map[name], lw=2)
        axes[0].fill_between(x, mean_acc - std_acc, mean_acc + std_acc, color=colors_map[name], alpha=0.15)
        
        # Dead Neurons
        mean_dead = np.mean(dead_neurons_per_seed, axis=0)
        std_dead = np.std(dead_neurons_per_seed, axis=0)
        axes[1].plot(x, mean_dead, label=name, color=colors_map[name], lw=2)
        axes[1].fill_between(x, mean_dead - std_dead, mean_dead + std_dead, color=colors_map[name], alpha=0.15)
        
        # Effective Rank
        mean_rank = np.mean(ranks_per_seed, axis=0)
        std_rank = np.std(ranks_per_seed, axis=0)
        axes[2].plot(x, mean_rank, label=name, color=colors_map[name], lw=2)
        axes[2].fill_between(x, mean_rank - std_rank, mean_rank + std_rank, color=colors_map[name], alpha=0.15)
        
    axes[0].set_title("Plasticity: Test Accuracy (Current Angle)")
    axes[0].set_xlabel("Epoch (Continuous rotation 0° to 180°)")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    axes[1].set_title("Capacity: Dead Neuron Fraction (h2)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dead Neuron Fraction")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    axes[2].set_title("Diversity: Effective Rank (h2)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Effective Rank")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "rotating_mnist_comparison.png"), dpi=200)
    plt.close()
    print("Rotating MNIST plots saved successfully.")

if __name__ == "__main__":
    run_experiment()
