import os
import pandas as pd

def summarize_suite(prefix, name, group_cols):
    base_dir = 'results'
    all_data = []
    for run_id in os.listdir(base_dir):
        if run_id.startswith(prefix) and os.path.isdir(os.path.join(base_dir, run_id)):
            parquet_path = os.path.join(base_dir, run_id, 'metrics.parquet')
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
                
                # Accuracies are duplicated across layers for the same task, so take layer=0
                if 'layer' in df.columns:
                    df = df[df['layer'] == 0]
                
                res = {}
                for col in group_cols:
                    if col in df.columns and col not in ['width', 'optimizer', 'lr', 'act_fn', 'lambda_rs']:
                        res[col] = df[col].iloc[0]
                    else:
                        # Extract from run_id if it's missing (e.g. width, optimizer, lr)
                        if col == 'width':
                            res[col] = int(run_id.split('width_')[1].split('_')[0])
                        elif col == 'optimizer':
                            res[col] = run_id.split('opt_')[1].split('_')[0]
                        elif col == 'lr':
                            res[col] = float(run_id.split('lr_')[1].split('_')[0])
                        elif col == 'method':
                            res[col] = df['method'].iloc[0] if 'method' in df.columns else run_id
                        elif col == 'act_fn':
                            if 'leaky_relu' in run_id:
                                res[col] = 'leaky_relu'
                            elif 'relu' in run_id:
                                res[col] = 'relu'
                            else:
                                res[col] = 'unknown'
                        elif col == 'lambda_rs':
                            res[col] = float(run_id.split('lam_')[1].split('_')[0])
                            
                res['avg_test_acc'] = df['test_acc'].mean()
                if 'prev_tasks_acc' in df.columns:
                    res['avg_prev_acc'] = df['prev_tasks_acc'].mean()
                else:
                    res['avg_prev_acc'] = float('nan')
                    
                res['seed'] = run_id.split('seed')[-1]
                all_data.append(res)
                
    if not all_data:
        return ""
        
    df_all = pd.DataFrame(all_data)
    
    # Average across seeds
    agg_dict = {'avg_test_acc': ['mean', 'std']}
    if not df_all['avg_prev_acc'].isna().all():
        agg_dict['avg_prev_acc'] = ['mean', 'std']
        
    df_agg = df_all.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Flatten columns
    df_agg.columns = [
        '_'.join(col).strip('_') for col in df_agg.columns.values
    ]
    
    md = f"## {name}\n\n"
    md += df_agg.to_markdown(index=False, floatfmt=".4f")
    md += "\n\n"
    return md

def main():
    out = ""
    out += summarize_suite("S4_3", "S4.3: Width Scaling (RS lambda=1.0)", ['width'])
    out += summarize_suite("S5_1", "S5.1: Rotating MNIST (MLP depth=2, width=256)", ['lambda_rs'])
    out += summarize_suite("S5_2", "S5.2: Split CIFAR-100 (ConvNet)", ['method'])
    out += summarize_suite("S5_3", "S5.3: Activation Functions (Permuted MNIST)", ['act_fn', 'lambda_rs'])
    out += summarize_suite("S6_1", "S6.1: Learning Rate Robustness", ['lr', 'method'])
    out += summarize_suite("S6_2", "S6.2: Optimizer Interaction (AdamW vs SGD)", ['optimizer'])
    
    with open('detailed_tables.md', 'w') as f:
        f.write(out)

if __name__ == '__main__':
    main()
