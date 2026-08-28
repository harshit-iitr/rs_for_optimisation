import pandas as pd
import glob

s52_paths = glob.glob('results/S5_2_*_seed*/metrics.parquet')
s52_records = []
for p in s52_paths:
    run = p.split('/')[-2]
    method = run.replace('S5_2_', '').split('_seed')[0]
    seed = int(run.split('_seed')[1])
    try:
        df = pd.read_parquet(p)
        task_df = df.groupby('task').mean(numeric_only=True)
        acc = task_df.loc[task_df.index[-2]:, 'test_acc'].mean()
        ret = task_df.loc[task_df.index[-2]:, 'prev_tasks_acc'].mean()
        s52_records.append({'method': method, 'seed': seed, 'acc': acc, 'ret': ret})
    except Exception as e:
        print(f'Error reading {p}: {e}')

s52_df = pd.DataFrame(s52_records)
if len(s52_df) > 0:
    agg2 = s52_df.groupby('method').agg(['mean', 'std'])
    for method, row in agg2.iterrows():
        print(f"{method}: Acc={row[('acc','mean')]:.4f}±{row[('acc','std')]:.4f}, Ret={row[('ret','mean')]:.4f}±{row[('ret','std')]:.4f}")
else:
    print("No S5.2 files found!")
