import torch
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import os
from tqdm import tqdm
from src.models.mlp import MLP
from src.data.permuted_mnist import PermutedMNIST
from src.methods.rs import compute_rs_penalty
from src.metrics.norms import compute_activation_radius

def run_timescale():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    lambdas = [1e-2, 0.1, 1.0]
    seeds = [1, 2, 3]
    
    results = []
    
    for lam in lambdas:
        for seed in seeds:
            print(f"Running lambda={lam} seed={seed}...")
            torch.manual_seed(seed)
            model = MLP(depth=3, width=1000).to(device)
            dataset = PermutedMNIST(n_tasks=50, device=device, seed=seed)
            
            optimizer = optim.SGD(model.parameters(), lr=1e-2)
            criterion = nn.CrossEntropyLoss()
            
            # Train for 50 tasks normally
            for t in range(50):
                x_train, y_train, _, _ = dataset.get_task_data(t)
                indices = torch.randperm(x_train.shape[0], device=device)
                
                model.train()
                for start_idx in range(0, x_train.shape[0], 256):
                    batch_idx = indices[start_idx:start_idx+256]
                    out, pre_acts, _ = model(x_train[batch_idx], return_activations=True)
                    loss = criterion(out, y_train[batch_idx])
                    
                    rs_loss = sum(compute_rs_penalty(h) for h in pre_acts)
                    loss += lam * rs_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    
            # Freeze data, perturb activations radially by +20%
            # We do this by scaling weights by 1.2
            with torch.no_grad():
                for layer in model.layers:
                    layer.weight.mul_(1.2)
                    layer.bias.mul_(1.2)
                    
            # Measure decay of u step-by-step
            # We will run 500 steps on the same task 49 data
            x_train, y_train, _, _ = dataset.get_task_data(49)
            
            for step in range(500):
                batch_idx = torch.randint(0, x_train.shape[0], (256,), device=device)
                
                out, pre_acts, _ = model(x_train[batch_idx], return_activations=True)
                
                # Log current u
                with torch.no_grad():
                    rad_mean, _ = compute_activation_radius(pre_acts[0]) # layer 0
                    u = rad_mean - math.sqrt(pre_acts[0].shape[-1])
                    results.append({
                        'lambda_rs': lam,
                        'seed': seed,
                        'step': step,
                        'u': float(u)
                    })
                    
                loss = criterion(out, y_train[batch_idx])
                rs_loss = sum(compute_rs_penalty(h) for h in pre_acts)
                loss += lam * rs_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_parquet('results/s4_2_timescale.parquet')
    print("Saved S4.2 timescale data.")

if __name__ == '__main__':
    run_timescale()
