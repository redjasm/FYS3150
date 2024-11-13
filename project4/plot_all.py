import os

# List of plot files to run
plot_files = [
    "plot_burning.py",
    "plot_critical_temp.py",
    "plot_energy_dist.py",
    "plot_phase_transition_improved.py",
    "plot_phase_transition.py",
    "plot_speedup.py"
]

# Run each plot file
for file in plot_files:
    print(f"Running {file}...")
    os.system(f"python scripts/{file}")

print("All plots generated successfully!")