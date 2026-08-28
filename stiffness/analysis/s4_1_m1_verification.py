import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

def main():
    lambdas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]
    seeds = [1, 2, 3]
    tasks_to_check = [10, 50, 100, 149] 
    
    records = []
    
    for lam in lambdas:
        for seed in seeds:
            run_id = f"S1_lam_{lam}_seed{seed}"
            path = f"results/{run_id}/metrics.parquet"
            if not os.path.exists(path):
                continue
                
            df = pd.read_parquet(path)
            for t in tasks_to_check:
                sub = df[df['task'] == t]
                for _, row in sub.iterrows():
                    records.append({
                        'lambda': lam,
                        'seed': seed,
                        'task': t,
                        'layer': row['layer'],
                        'radial_excess': row['radial_excess']
                    })
                    
    if not records:
        print("No records found.")
        return
        
    df = pd.DataFrame(records)
    df['abs_radial_excess'] = df['radial_excess'].abs()
    
    # log-log fit
    df['log_lambda'] = np.log10(df['lambda'])
    # avoid log(0)
    df = df[df['abs_radial_excess'] > 1e-8]
    df['log_excess'] = np.log10(df['abs_radial_excess'])
    
    print("S4.1: M1 Verification (Predicted slope = -1)")
    print("-" * 50)
    
    for t in tasks_to_check:
        print(f"\n--- Task {t} ---")
        sub_t = df[df['task'] == t]
        
        for l in sorted(sub_t['layer'].unique()):
            sub_l = sub_t[sub_t['layer'] == l]
            if len(sub_l) < 3:
                continue
                
            X = sm.add_constant(sub_l['log_lambda'])
            model = sm.OLS(sub_l['log_excess'], X).fit()
            slope = model.params['log_lambda']
            conf = model.conf_int(alpha=0.05).loc['log_lambda']
            
            print(f"Layer {l}: slope = {slope:.3f} [95% CI: {conf[0]:.3f}, {conf[1]:.3f}]")

if __name__ == '__main__':
    main()
