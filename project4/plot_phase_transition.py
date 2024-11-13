import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Read data
data = pd.read_csv('data/phase_transition_data.csv')

# Create plots for each observable
observables = {
    'epsilon': ('Energy per spin [J]', 'ϵ'),
    'm_abs': ('Magnetization per spin', '|m|'),
    'Cv': ('Specific heat per spin [k_B]', 'C_V/N'),
    'chi': ('Susceptibility per spin', 'χ/N')
}

for obs, (ylabel, title) in observables.items():
    plt.figure(figsize=(10, 6))
    
    for L in data['L'].unique():
        subset = data[data['L'] == L]
        plt.plot(subset['T'], subset[obs], 'o-', label=f'L = {L}')
    
    plt.xlabel('Temperature [J/k_B]')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'figures/phase_transition_{obs}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Find critical temperatures from susceptibility peaks
Tc_values = []
for L in sorted(data['L'].unique()):
    subset = data[data['L'] == L]
    Tc_idx = subset['chi'].idxmax()
    Tc = subset.loc[Tc_idx, 'T']
    Tc_values.append((L, Tc))

print("\nCritical temperatures:")
for L, Tc in Tc_values:
    print(f"L = {L:3d}: Tc = {Tc:.4f} J/k_B")