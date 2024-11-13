import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Create figures directory
os.makedirs('figures', exist_ok=True)

# Read data
data = pd.read_csv('data/phase_transition_improved.csv')

# Group by L and T, calculate mean of observables
grouped = data.groupby(['L', 'T']).agg({
    'chi': 'mean',
    'Cv': 'mean',
    'epsilon': 'mean',
    'm_abs': 'mean'
}).reset_index()

# Find critical temperatures from susceptibility peaks
Tc_values = []
for L in sorted(data['L'].unique()):
    L_data = grouped[grouped['L'] == L]
    max_idx = L_data['chi'].idxmax()
    Tc = L_data.loc[max_idx, 'T']
    Tc_values.append(Tc)

Tc_values = np.array(Tc_values)
L_values = np.sort(data['L'].unique())
L_inv = 1/L_values

# Simple linear regression
lin = linregress(L_inv, Tc_values)
Tc_infty = lin.intercept
dTc = lin.intercept_stderr

# Onsager's solution
Tc_onsager = 2.269

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(L_inv, Tc_values, 'o', 
        label='Computed Tc(L)', color='#377eb8', markersize=8)
plt.plot(L_inv, lin.slope*L_inv + lin.intercept, '-', 
         label='Linear fit', color='#4daf4a')

# Add Onsager's solution
plt.plot(0, Tc_onsager, '*', label="Onsager's solution",
         color='red', markersize=15)

plt.xlabel('1/L')
plt.ylabel('Tc [J/kB]')
plt.title('Critical Temperature vs System Size (Improved Analysis)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-0.001, 0.026)

plt.tight_layout()
plt.savefig('figures/critical_temp_scaling_improved.pdf')
plt.close()

# Print results
print(f"\nResults from linear regression:")
print(f"Tc(L->infinity) = {Tc_infty:.4f} ± {dTc:.4f} J/kB")
print(f"Slope = {lin.slope:.4f} ± {lin.stderr:.4f}")
print(f"\nComparison with Onsager's solution:")
print(f"Onsager's Tc = {Tc_onsager:.4f} J/kB")
print(f"Absolute error: {abs(Tc_infty-Tc_onsager):.4f}")
print(f"Relative error: {abs(Tc_infty-Tc_onsager)/Tc_onsager:.4%}")

# Print individual Tc values
print("\nCritical temperatures for each lattice size:")
for L, Tc in zip(L_values, Tc_values):
    print(f"L = {L:3d}: Tc = {Tc:.4f} J/kB")