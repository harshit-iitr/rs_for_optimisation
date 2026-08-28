import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def analyze_b4():
    base_dir = 'results'
    all_data = []
    
    for run_id in os.listdir(base_dir):
        if not run_id.startswith('R2_B4_'): continue
        path = os.path.join(base_dir, run_id, 'metrics.parquet')
        if not os.path.exists(path): continue
        
        df = pd.read_parquet(path)
        all_data.append(df)
        
    if not all_data:
        print("No B4 data found.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    
    tasks_to_plot = [9, 24, 49]
    layers = df['layer'].unique()
    
    plt.figure(figsize=(15, 5))
    
    for i, t in enumerate(tasks_to_plot):
        plt.subplot(1, 3, i+1)
        df_t = df[df['task'] == t]
        
        for l in sorted(layers):
            df_l = df_t[df_t['layer'] == l]
            # Average across seeds
            grouped = df_l.groupby('lambda_rs')['radial_excess'].apply(lambda x: np.mean(np.abs(x))).reset_index()
            grouped = grouped[grouped['lambda_rs'] > 0]
            
            x = np.log10(grouped['lambda_rs'])
            y = np.log10(grouped['radial_excess'])
            
            plt.plot(grouped['lambda_rs'], grouped['radial_excess'], marker='o', label=f'Layer {l}')
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            ci = 1.96 * std_err
            print(f"Task {t+1}, Layer {l}: log-log slope = {slope:.2f} ± {ci:.2f}")
            
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('|Radial Excess|')
        plt.title(f'End of Task {t+1}')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig('results/b4_lambda_law.png')
    print("Saved plot to results/b4_lambda_law.png")

if __name__ == '__main__':
    analyze_b4()
