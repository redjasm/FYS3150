import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Style settings
plt.style.use('default')
colors = ['#377eb8', '#e41a1c']

# Read data
data_T1 = pd.read_csv('data/energy_dist_T1.0.csv')
data_T24 = pd.read_csv('data/energy_dist_T2.4.csv')

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot T = 1.0 distribution
ax1.hist(data_T1['energy_per_spin'], bins=50, density=True, 
         alpha=0.7, color=colors[0], label='Histogram')
# Add KDE
kde1 = stats.gaussian_kde(data_T1['energy_per_spin'])
x1 = np.linspace(data_T1['energy_per_spin'].min(), 
                data_T1['energy_per_spin'].max(), 200)
ax1.plot(x1, kde1(x1), 'k-', lw=2, label='KDE')
ax1.set_title('T = 1.0 J/k_B')
ax1.set_xlabel('Energy per spin [J]')
ax1.set_ylabel('Probability density')
ax1.legend()

# Plot T = 2.4 distribution
ax2.hist(data_T24['energy_per_spin'], bins=50, density=True, 
         alpha=0.7, color=colors[1], label='Histogram')
# Add KDE
kde2 = stats.gaussian_kde(data_T24['energy_per_spin'])
x2 = np.linspace(data_T24['energy_per_spin'].min(), 
                data_T24['energy_per_spin'].max(), 200)
ax2.plot(x2, kde2(x2), 'k-', lw=2, label='KDE')
ax2.set_title('T = 2.4 J/k_B')
ax2.set_xlabel('Energy per spin [J]')
ax2.legend()

# Calculate statistics
stats_T1 = data_T1['energy_per_spin'].describe()
stats_T24 = data_T24['energy_per_spin'].describe()
var_T1 = stats_T1['std']**2
var_T24 = stats_T24['std']**2

# Add text boxes with statistics
text_T1 = f'μ = {stats_T1["mean"]:.3f}J\nσ² = {var_T1:.3e}J²'
text_T24 = f'μ = {stats_T24["mean"]:.3f}J\nσ² = {var_T24:.3e}J²'
ax1.text(0.05, 0.95, text_T1, transform=ax1.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))
ax2.text(0.05, 0.95, text_T24, transform=ax2.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/energy_distributions.pdf')
plt.close()

# Print detailed statistics
print("\nStatistics for T = 1.0 J/k_B:")
print(stats_T1)
print("\nStatistics for T = 2.4 J/k_B:")
print(stats_T24)
print("\nVariance ratio (T2.4/T1.0):", var_T24/var_T1)