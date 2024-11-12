import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create directories if they don't exist
os.makedirs('figures', exist_ok=True)
os.makedirs('data/output', exist_ok=True)

# Read timing data
data = pd.read_csv('data/output/parallel_timing.csv')

plt.figure(figsize=(10, 6))

# Plot measured speedup
plt.plot(data['threads'], data['speedup'], 'o-', label='Measured speedup', 
         color='#377eb8', linewidth=2, markersize=8)

# Plot ideal speedup (1:1 line)
x = np.array([1, data['threads'].max()])
plt.plot(x, x, '--', color='gray', label='Ideal speedup', alpha=0.7)

plt.xlabel('Number of threads')
plt.ylabel('Speedup factor')
plt.title('OpenMP Parallelization Speedup for Ising Model')
plt.grid(True, alpha=0.3)
plt.legend()

# Add annotations for efficiency
for _, row in data.iterrows():
    if row['threads'] > 1:
        efficiency = row['speedup'] / row['threads'] * 100
        plt.annotate(f'{efficiency:.1f}% efficient', 
                    (row['threads'], row['speedup']),
                    xytext=(10, 10), textcoords='offset points')

plt.tight_layout()
plt.savefig('figures/parallel_speedup.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Print the timing data
print("\nTiming Results:")
print(data.to_string(index=False))