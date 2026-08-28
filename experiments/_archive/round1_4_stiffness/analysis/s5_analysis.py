import os
import pandas as pd
import glob
import numpy as np

def analyze_s5():
    print("--- S5.1 Rotating MNIST ---")
    s51_paths = glob.glob('results/S5_1_rot_*/metrics.parquet')
    s51_records = []
    for p in s51_paths:
        run = p.split('/')[-2]
        lam_str = run.split('_lam_')[1].split('_seed')[0]
        lam = float(lam_str)
        seed = int(run.split('_seed')[1])
        try:
            df = pd.read_parquet(p)
            task_df = df.groupby('task').mean(numeric_only=True)
            acc = task_df.loc[task_df.index[-20]:, 'test_acc'].mean()
            ret = task_df.loc[task_df.index[-20]:, 'prev_tasks_acc'].mean()
            s51_records.append({'lam': lam, 'seed': seed, 'acc': acc, 'ret': ret})
        except Exception as e:
            pass
            
    s51_df = pd.DataFrame(s51_records)
    if len(s51_df) > 0:
        agg1 = s51_df.groupby('lam').agg(['mean', 'std'])
        for lam, row in agg1.iterrows():
            print(f"λ={lam}: Acc={row[('acc','mean')]:.4f}±{row[('acc','std')]:.4f}, Ret={row[('ret','mean')]:.4f}±{row[('ret','std')]:.4f}")
            
    print("\n--- S5.2 Split CIFAR-100 ---")
    s52_paths = glob.glob('results/S5_2_*/metrics.parquet')
    s52_records = []
    for p in s52_paths:
        run = p.split('/')[-2]
        if '_seed' not in run:
            continue
        method = run.replace('S5_2_', '').split('_seed')[0]
        seed = int(run.split('_seed')[1])
        try:
            df = pd.read_parquet(p)
            task_df = df.groupby('task').mean(numeric_only=True)
            acc = task_df.loc[task_df.index[-2]:, 'test_acc'].mean()
            ret = task_df.loc[task_df.index[-2]:, 'prev_tasks_acc'].mean()
            s52_records.append({'method': method, 'seed': seed, 'acc': acc, 'ret': ret})
        except:
            pass
            
    s52_df = pd.DataFrame(s52_records)
    if len(s52_df) > 0:
        agg2 = s52_df.groupby('method').agg(['mean', 'std'])
        for method, row in agg2.iterrows():
            print(f"{method}: Acc={row[('acc','mean')]:.4f}±{row[('acc','std')]:.4f}, Ret={row[('ret','mean')]:.4f}±{row[('ret','std')]:.4f}")
            
    print("\n--- S5.3 ReLU vs LeakyReLU ---")
    s53_paths = glob.glob('results/S5_3_*/metrics.parquet')
    s53_records = []
    for p in s53_paths:
        run = p.split('/')[-2]
        act = 'leaky_relu' if 'leaky_relu' in run else 'relu'
        lam_str = run.split('_lam_')[1].split('_seed')[0]
        lam = float(lam_str)
        seed = int(run.split('_seed')[1])
        try:
            df = pd.read_parquet(p)
            task_df = df.groupby('task').mean(numeric_only=True)
            acc = task_df.loc[task_df.index[-20]:, 'test_acc'].mean()
            ret = task_df.loc[task_df.index[-20]:, 'prev_tasks_acc'].mean()
            s53_records.append({'act': act, 'lam': lam, 'seed': seed, 'acc': acc, 'ret': ret})
        except:
            pass
            
    s53_df = pd.DataFrame(s53_records)
    if len(s53_df) > 0:
        agg3 = s53_df.groupby(['act', 'lam']).agg(['mean', 'std'])
        for (act, lam), row in agg3.iterrows():
            print(f"{act}, λ={lam}: Acc={row[('acc','mean')]:.4f}±{row[('acc','std')]:.4f}, Ret={row[('ret','mean')]:.4f}±{row[('ret','std')]:.4f}")
            
if __name__ == '__main__':
    analyze_s5()
