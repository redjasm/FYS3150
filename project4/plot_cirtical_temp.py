import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Data from our simulations
L = np.array([40, 60, 80, 100])
Tc = np.array([2.3184, 2.2658, 2.2763, 2.1500])

# Calculate 1/L for x-axis
L_inv = 1/L

# Perform linear regression
lin = linregress(L_inv, Tc)
a = lin.slope
Tc_infty = lin.intercept

# Calculate uncertainties
da = lin.stderr
dTc = lin.intercept_stderr

# Onsager's analytical result
Tc_onsager = 2.269

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(L_inv, Tc, 'o', label='Computed Tc(L)', color='#377eb8', markersize=8)
plt.plot(L_inv, a*L_inv + Tc_infty, '-', 
         label='Linear fit', color='#4daf4a', alpha=0.7)

# Plot range for visualization
x_fit = np.linspace(0, 0.03, 100)
plt.plot(x_fit, a*x_fit + Tc_infty, '--', color='#4daf4a', alpha=0.3)

# Add point for Onsager's solution
plt.plot(0, Tc_onsager, '*', label="Onsager's solution",
         color='red', markersize=15)

plt.xlabel('1/L')
plt.ylabel('Tc [J/kB]')
plt.title('Critical Temperature vs System Size')
plt.legend()
plt.grid(True, alpha=0.3)

# Adjust plot range to show x=0
plt.xlim(-0.001, 0.026)

plt.tight_layout()
plt.savefig('figures/critical_temp_scaling.pdf')
plt.close()

# Print results
print(f"\nResults from linear regression:")
print(f"Tc(L->infinity) = {Tc_infty:.4f} ± {dTc:.4f} J/kB")
print(f"Slope a = {a:.4f} ± {da:.4f}")
print(f"\nComparison with Onsager's solution:")
print(f"Onsager's Tc = {Tc_onsager:.4f} J/kB")
print(f"Absolute error: {abs(Tc_infty-Tc_onsager):.4f}")
print(f"Relative error: {abs(Tc_infty-Tc_onsager)/Tc_onsager:.4%}")