import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
import math
import copy
from tqdm import tqdm

from src.models.mlp import MLP
from src.models.convnet import ConvNet
from src.data.permuted_mnist import PermutedMNIST
from src.data.rotating_mnist import RotatingMNIST
from src.data.split_cifar100 import SplitCIFAR100
from src.methods.rs import compute_rs_penalty
from src.metrics.phi_rad import compute_phi_rad_tilde
from src.metrics.rank import compute_effective_rank, compute_stable_rank
from src.metrics.neurons import compute_dead_fraction, compute_dormant_fraction
from src.metrics.norms import compute_activation_radius, compute_weight_norm, compute_grad_norm
from src.metrics.readiness import compute_readiness

from src.methods.baselines import apply_shrink_and_perturb, compute_l2_init_penalty, compute_er_penalty
from src.methods.cbp import CBP
from src.methods.redo import ReDo
from src.methods.ewc import EWC
from src.methods.si import SI
from src.methods.mas import MAS

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='bp', choices=['bp', 'l2', 'l2_init', 'ln', 'ln_l2', 'sp', 'redo', 'er', 'l2_er', 'rs', 'ewc', 'si', 'mas', 'rs_ewc', 'bp_iso'], help="Continual learning method")
    parser.add_argument('--lambda_rs', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--hard_projection', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_tasks', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log_within_task', action='store_true')
    parser.add_argument('--dataset', type=str, default='permuted_mnist')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--width', type=int, default=1000)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--act_fn', type=str, default='relu')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--run_id', type=str, default='debug')
    parser.add_argument('--track_drift', action='store_true')
    parser.add_argument('--ewc_lambda', type=float, default=1000.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    act_class = nn.ReLU if args.act_fn == 'relu' else nn.LeakyReLU
    use_ln = args.method in ['ln', 'ln_l2']
    if args.model == 'mlp':
        model = MLP(use_ln=use_ln, depth=args.depth, width=args.width, act_fn=act_class).to(device)
    else:
        model = ConvNet(use_ln=use_ln, act_fn=act_class).to(device)
    
    initial_model = None
    if args.method == 'l2_init':
        initial_model = copy.deepcopy(model)
        initial_model.eval()
        
    cbp = None
        
    redo = None
    if args.method == 'redo':
        redo = ReDo(model)

    ewc = EWC(model) if args.method in ['ewc', 'rs_ewc'] else None
    si = SI(model) if args.method == 'si' else None
    mas = MAS(model) if args.method == 'mas' else None

    if args.dataset == 'permuted_mnist':
        dataset = PermutedMNIST(n_tasks=args.n_tasks, device=device, seed=args.seed)
    elif args.dataset == 'rotating_mnist':
        dataset = RotatingMNIST(n_tasks=args.n_tasks, device=device, seed=args.seed)
    elif args.dataset == 'split_cifar100':
        dataset = SplitCIFAR100(n_tasks=args.n_tasks, device=device, seed=args.seed)
    
    probe_batch_size = 2000 if args.dataset != 'split_cifar100' else 256
    
    shrinkage = None
    if args.method == 'bp_iso':
        import json
        with open('results/iso_shrinkage.json', 'r') as f:
            shrinkage = json.load(f)

    wd = args.weight_decay if args.method in ['l2', 'ln_l2', 'l2_er'] else 0.0
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=wd)
    elif args.optimizer == 'sgd_momentum':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    metrics_records = []
    
    os.makedirs(f'results/{args.run_id}', exist_ok=True)
    
    # (removed unconditional probe_batch_size = 2000)

    past_task_tests = []
    
    if args.track_drift:
        drift_probe_x, _, _, _ = dataset.get_task_data(0)
        drift_probe_x = drift_probe_x[:1000].clone().to(device)
        past_acts = None

    for t in tqdm(range(args.n_tasks), desc=f"Run {args.run_id}"):
        x_train, y_train, x_test, y_test = dataset.get_task_data(t)
        past_task_tests.append((x_test, y_test))
        
        # Init accuracy
        model.eval()
        with torch.no_grad():
            init_logits = model(x_train[:probe_batch_size], hard_projection=args.hard_projection)
            init_acc = (init_logits.argmax(dim=1) == y_train[:probe_batch_size]).float().mean().item()
            
        # Setup within-task logging
        log_this_task = args.log_within_task and (t < 3 or t >= args.n_tasks - 3)
        within_task_records = []
        step_idx = 0

        # Training
        n_samples = x_train.shape[0]
        
        model.train()
        last_update_norms = {l: 0.0 for l in range(len(model.layers))}
        
        for ep in range(args.epochs):
            indices = torch.randperm(n_samples, device=device)
            for start_idx in range(0, n_samples, 256):
                
                if log_this_task and step_idx % 25 == 0:
                    model.eval()
                    with torch.no_grad():
                        t_logits = model(x_test[:probe_batch_size], hard_projection=args.hard_projection)
                        t_acc = (t_logits.argmax(dim=1) == y_test[:probe_batch_size]).float().mean().item()
                    within_task_records.append({'run_id': args.run_id, 'task': t, 'epoch': ep, 'step': step_idx, 'test_acc': t_acc})
                    model.train()

                batch_idx = indices[start_idx:start_idx+256]
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                optimizer.zero_grad()
                out, pre_acts, post_acts = model(x_batch, return_activations=True, hard_projection=args.hard_projection)
                loss = criterion(out, y_batch)
                
                if args.method == 'l2_init':
                    loss += 0.01 * compute_l2_init_penalty(model, initial_model)
                    
                if args.method in ['er', 'l2_er']:
                    for h in pre_acts:
                        loss += 0.01 * compute_er_penalty(h)
                        
                if ewc is not None:
                    loss += args.ewc_lambda * ewc.penalty()
                if si is not None:
                    loss += 1.0 * si.penalty()
                if mas is not None:
                    loss += 1.0 * mas.penalty()
                
                if args.lambda_rs > 0 and not args.hard_projection:
                    rs_loss = sum(compute_rs_penalty(h) for h in pre_acts)
                    loss += args.lambda_rs * rs_loss
                    
                loss.backward()
            
                if args.method == 'bp_iso' and shrinkage is not None:
                    t_str = str(t)
                    if t_str in shrinkage:
                        for l, layer in enumerate(model.layers):
                            l_str = str(l)
                            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                                if l_str in shrinkage[t_str]:
                                    s = shrinkage[t_str][l_str]
                                    layer.weight.grad.data.mul_(s)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                
                is_last_step = (ep == args.epochs - 1) and (start_idx + 256 >= n_samples)
                if is_last_step:
                    old_weights = [layer.weight.clone().detach() for layer in model.layers]
                    
                if si is not None:
                    old_weights_dict = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
                    
                optimizer.step()
                
                if si is not None:
                    si.update_W(old_weights_dict)
                
                if is_last_step:
                    for l, layer in enumerate(model.layers):
                        update_norm = torch.norm(layer.weight.detach() - old_weights[l]).item()
                        last_update_norms[l] = update_norm
                
                pass
                    
                if redo is not None:
                    redo.check_and_recycle(post_acts)
                    
                step_idx += 1
                
        if log_this_task:
            wt_df = pd.DataFrame(within_task_records)
            wt_file = f'results/{args.run_id}/within_task.csv'
            wt_df.to_csv(wt_file, mode='a', header=not os.path.exists(wt_file), index=False)
            
        # End of task perturbation
        if args.method == 'sp':
            apply_shrink_and_perturb(model)
            
        if ewc is not None:
            model.zero_grad()
            out = model(x_train[:probe_batch_size])
            loss_ewc = criterion(out, y_train[:probe_batch_size])
            loss_ewc.backward()
            ewc.update_fisher()
            model.zero_grad()
            
        if si is not None:
            si.update_omega()
            
        if mas is not None:
            model.zero_grad()
            out = model(x_train[:probe_batch_size])
            l2_norm = out.pow(2).sum(dim=1).mean()
            l2_norm.backward()
            mas.update_omega()
            model.zero_grad()
            
        # End of task probe evaluation
        actual_probe_size = min(probe_batch_size, len(x_test))
        probe_x = x_test[:actual_probe_size]
        probe_y = y_test[:actual_probe_size]
        
        # 1. Readiness (network-wide)
        m = 8
        micro_size = actual_probe_size // m
        # Collect gradients over microbatches per layer
        g_microbatches_per_layer = {l: [] for l in range(len(model.layers))}
        for i in range(m):
            start = i * micro_size
            end = start + micro_size
            mb_x = probe_x[start:end]
            mb_y = probe_y[start:end]
            
            if len(mb_x) == 0:
                continue
                
            model.zero_grad()
            mb_out = model(mb_x, hard_projection=args.hard_projection)
            mb_loss = criterion(mb_out, mb_y)
            mb_loss.backward()
            
            for l, layer in enumerate(model.layers):
                if hasattr(layer, 'weight') and layer.weight.grad is not None:
                    g_microbatches_per_layer[l].append(layer.weight.grad.detach().clone().view(-1))
                    
        readiness_per_layer = {}
        for l in range(len(model.layers)):
            if len(g_microbatches_per_layer[l]) > 0:
                readiness_per_layer[l] = compute_readiness(g_microbatches_per_layer[l])
            else:
                readiness_per_layer[l] = float('nan')
        
        # 2. Per-layer metrics & phi_rad_tilde
        model.zero_grad()
        out_probe, pre_probe, post_probe = model(probe_x, return_activations=True, hard_projection=args.hard_projection)
        loss_probe = criterion(out_probe, probe_y)
        
        g_tasks = torch.autograd.grad(loss_probe, pre_probe, retain_graph=True)
        
        loss_probe.backward()
        
        layer_metrics = []
        for l in range(len(pre_probe)):
            h = pre_probe[l].detach()
            h_post = post_probe[l].detach()
            g_task = g_tasks[l].detach()
            weight = model.layers[l].weight
            
            phi_rad = compute_phi_rad_tilde(h, g_task)
            h_norm = torch.norm(h, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            h_hat = h / h_norm
            g_rad = torch.sum(g_task * h_hat, dim=-1, keepdim=True) * h_hat
            g_rad_norm = torch.sqrt(torch.sum(g_rad**2, dim=-1)).mean().item()
            g_norm_task = torch.norm(g_task, p=2, dim=-1).mean().item()
            
            rad_mean, rad_std = compute_activation_radius(h)
            radial_excess = rad_mean - math.sqrt(h.shape[-1])
            eff_rank = compute_effective_rank(h)
            stable_rank = compute_stable_rank(h)
            dead_frac = compute_dead_fraction(h_post)
            dormant_frac = compute_dormant_fraction(h_post)
            w_norm = compute_weight_norm(weight)
            g_norm = compute_grad_norm(weight.grad)
            u_norm = last_update_norms[l]
            
            layer_metrics.append({
                'layer': l,
                'phi_rad_tilde': phi_rad,
                'radial_excess': radial_excess,
                'radius_mean': rad_mean,
                'radius_std': rad_std,
                'eff_rank': eff_rank,
                'stable_rank': stable_rank,
                'dead_frac': dead_frac,
                'dormant_frac': dormant_frac,
                'weight_norm': w_norm,
                'grad_norm': g_norm,
                'update_norm': u_norm,
                'g_rad_norm': g_rad_norm,
                'g_norm_task': g_norm_task,
            })
            
        # 2.5 Track drift
        if args.track_drift:
            model.eval()
            with torch.no_grad():
                _, curr_acts, _ = model(drift_probe_x, return_activations=True, hard_projection=args.hard_projection)
            
            for l in range(len(curr_acts)):
                h_curr = curr_acts[l].detach()
                lm = next(item for item in layer_metrics if item["layer"] == l)
                if past_acts is not None:
                    h_past = past_acts[l]
                    cos_sim = torch.nn.functional.cosine_similarity(h_curr.flatten(1), h_past.flatten(1), dim=1).mean().item()
                    rel_drift = (torch.norm(h_curr - h_past) / (torch.norm(h_past) + 1e-8)).item()
                    
                    U_curr, _, _ = torch.svd(h_curr.flatten(1).t())
                    U_past, _, _ = torch.svd(h_past.flatten(1).t())
                    k = min(50, U_curr.shape[1], U_past.shape[1])
                    cos_angles = torch.svd(U_curr[:, :k].t() @ U_past[:, :k])[1]
                    overlap = cos_angles.mean().item()
                    
                    lm['drift_cos_sim'] = cos_sim
                    lm['drift_rel'] = rel_drift
                    lm['drift_overlap'] = overlap
                    
                    delta_h = h_curr - h_past
                    h_past_norm = torch.norm(h_past, p=2, dim=1, keepdim=True).clamp(min=1e-8)
                    h_past_hat = h_past / h_past_norm
                    delta_h_rad = torch.sum(delta_h * h_past_hat, dim=1, keepdim=True) * h_past_hat
                    delta_h_tan = delta_h - delta_h_rad
                    
                    lm['drift_rad'] = (torch.norm(delta_h_rad) / (torch.norm(h_past) + 1e-8)).item()
                    lm['drift_tan'] = (torch.norm(delta_h_tan) / (torch.norm(h_past) + 1e-8)).item()
                else:
                    lm['drift_cos_sim'] = float('nan')
                    lm['drift_rel'] = float('nan')
                    lm['drift_overlap'] = float('nan')
                    lm['drift_rad'] = float('nan')
                    lm['drift_tan'] = float('nan')
            past_acts = [h.clone() for h in curr_acts]
        else:
            for lm in layer_metrics:
                lm['drift_cos_sim'] = float('nan')
                lm['drift_rel'] = float('nan')
                lm['drift_overlap'] = float('nan')
                lm['drift_rad'] = float('nan')
                lm['drift_tan'] = float('nan')
            
        # 3. Accuracy metrics
        model.eval()
        with torch.no_grad():
            test_logits = model(probe_x, hard_projection=args.hard_projection)
            test_acc = (test_logits.argmax(dim=1) == probe_y).float().mean().item()
            train_logits = model(x_train[:probe_batch_size], hard_projection=args.hard_projection)
            train_acc = (train_logits.argmax(dim=1) == y_train[:probe_batch_size]).float().mean().item()
            
            # Compute accuracy on all previously seen tasks
            prev_accs = []
            for px, py in past_task_tests:
                logits = model(px[:probe_batch_size], hard_projection=args.hard_projection)
                acc = (logits.argmax(dim=1) == py[:probe_batch_size]).float().mean().item()
                prev_accs.append(acc)
            prev_tasks_acc = sum(prev_accs) / len(prev_accs) if prev_accs else test_acc
            task_0_acc = prev_accs[0] if prev_accs else test_acc
            
        first_epoch_gain = train_acc - init_acc
        
        # Combine and log
        for lm in layer_metrics:
            metrics_records.append({
                'run_id': args.run_id,
                'task': t,
                'layer': lm['layer'],
                'method': args.method,
                'lambda_rs': args.lambda_rs,
                'hard_projection': args.hard_projection,
                'seed': args.seed,
                'phi_rad_tilde': lm['phi_rad_tilde'],
                'radial_excess': lm['radial_excess'],
                'radius_mean': lm['radius_mean'],
                'radius_std': lm['radius_std'],
                'eff_rank': lm['eff_rank'],
                'stable_rank': lm['stable_rank'],
                'dead_frac': lm['dead_frac'],
                'dormant_frac': lm['dormant_frac'],
                'weight_norm': lm['weight_norm'],
                'grad_norm': lm['grad_norm'],
                'update_norm': lm['update_norm'],
                'g_rad_norm': lm['g_rad_norm'],
                'g_norm_task': lm['g_norm_task'],
                'drift_cos_sim': lm['drift_cos_sim'],
                'drift_rel': lm['drift_rel'],
                'drift_overlap': lm['drift_overlap'],
                'drift_rad': lm['drift_rad'],
                'drift_tan': lm['drift_tan'],
                'readiness': readiness_per_layer.get(lm['layer'], float('nan')),
                'train_acc': train_acc,
                'test_acc': test_acc,
                'prev_tasks_acc': prev_tasks_acc,
                'task_0_acc': task_0_acc,
                'first_epoch_gain': first_epoch_gain
            })
            
    df = pd.DataFrame(metrics_records)
    df.to_parquet(f'results/{args.run_id}/metrics.parquet')

if __name__ == '__main__':
    main()
