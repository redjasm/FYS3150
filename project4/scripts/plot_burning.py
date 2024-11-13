import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set figure size and style
plt.figure(figsize=(12, 8))
plt.style.use('default')

# Define colors
colors = ['#377eb8', '#4daf4a', '#e41a1c', '#984ea3']

# Data directory
data_dir = "data"

try:
    # Load data with explicit paths
    data_files = {
        'T1_rand': os.path.join(data_dir, "burnin_T1.0_random.csv"),
        'T1_ord': os.path.join(data_dir, "burnin_T1.0_ordered.csv"),
        'T24_rand': os.path.join(data_dir, "burnin_T2.4_random.csv"),
        'T24_ord': os.path.join(data_dir, "burnin_T2.4_ordered.csv")
    }

    # Check if all files exist
    for name, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {path}")
    
    # Load data
    data_T1_rand = pd.read_csv(data_files['T1_rand'])
    data_T1_ord = pd.read_csv(data_files['T1_ord'])
    data_T24_rand = pd.read_csv(data_files['T24_rand'])
    data_T24_ord = pd.read_csv(data_files['T24_ord'])

    # Create main plot
    plt.plot(data_T1_rand['cycle'], data_T1_rand['instant_energy'], 
             '-', color=colors[0], alpha=0.4, linewidth=1.0)
    plt.plot(data_T1_ord['cycle'], data_T1_ord['instant_energy'], 
             '-', color=colors[1], alpha=0.4, linewidth=1.0)
    plt.plot(data_T24_rand['cycle'], data_T24_rand['instant_energy'], 
             '-', color=colors[2], alpha=0.4, linewidth=1.0)
    plt.plot(data_T24_ord['cycle'], data_T24_ord['instant_energy'], 
             '-', color=colors[3], alpha=0.4, linewidth=1.0)

    # Plot means
    plt.plot(data_T1_rand['cycle'], data_T1_rand['cumulative_energy'], 
             '-', linewidth=2.0, color=colors[0], 
             label='T=1.0 J/k_B, unordered')
    plt.plot(data_T1_ord['cycle'], data_T1_ord['cumulative_energy'], 
             '-', linewidth=2.0, color=colors[1], 
             label='T=1.0 J/k_B, ordered')
    plt.plot(data_T24_rand['cycle'], data_T24_rand['cumulative_energy'], 
             '-', linewidth=2.0, color=colors[2], 
             label='T=2.4 J/k_B, unordered')
    plt.plot(data_T24_ord['cycle'], data_T24_ord['cumulative_energy'], 
             '-', linewidth=2.0, color=colors[3], 
             label='T=2.4 J/k_B, ordered')

    plt.xscale('log')
    plt.xlabel('Monte Carlo cycles')
    plt.ylabel('Energy per spin [J]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Burn-in Study: Energy vs Monte Carlo Cycles')
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('figures/burnin_study.pdf', dpi=300, bbox_inches='tight')
    plt.close()

except FileNotFoundError as e:
    print(f"Error: {str(e)}")
    print("Make sure to run the burning_study program first to generate the data files.")
    print("Expected data files in directory 'data/':")
    for name, path in data_files.items():
        print(f"  - {os.path.basename(path)}")