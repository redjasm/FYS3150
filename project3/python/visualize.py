import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_error_analysis(n_steps):
    plt.figure(figsize=(12, 8))
    for n in n_steps:
        data = np.genfromtxt(f'build/error_{n}.csv', delimiter=',', skip_header=1)
        t, fe_error, rk4_error = data.T
        
        plt.subplot(121)
        plt.loglog(t, fe_error, label=f'n={n}')
        plt.xlabel('Time (μs)')
        plt.ylabel('Relative Error')
        plt.title('Forward Euler Error')
        plt.legend()
        
        plt.subplot(122)
        plt.loglog(t, rk4_error, label=f'n={n}')
        plt.xlabel('Time (μs)')
        plt.ylabel('Relative Error')
        plt.title('RK4 Error')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('build/figures/error_analysis.pdf')
    plt.close()

def plot_phase_space(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    t = data[:, 0]
    
    # Extract positions and velocities for both particles
    x1, y1, z1, vx1, vy1, vz1 = data[:, 1:7].T
    x2, y2, z2, vx2, vy2, vz2 = data[:, 7:13].T
    
    plt.figure(figsize=(12, 10))
    
    # Phase space plots for particle 1
    plt.subplot(221)
    plt.plot(x1, vx1, label='Particle 1')
    plt.xlabel('x (μm)')
    plt.ylabel('vx (μm/μs)')
    plt.title('x-vx Phase Space')
    plt.legend()
    
    plt.subplot(222)
    plt.plot(z1, vz1, label='Particle 1')
    plt.xlabel('z (μm)')
    plt.ylabel('vz (μm/μs)')
    plt.title('z-vz Phase Space')
    plt.legend()
    
    # Phase space plots for particle 2
    plt.subplot(223)
    plt.plot(x2, vx2, label='Particle 2')
    plt.xlabel('x (μm)')
    plt.ylabel('vx (μm/μs)')
    plt.legend()
    
    plt.subplot(224)
    plt.plot(z2, vz2, label='Particle 2')
    plt.xlabel('z (μm)')
    plt.ylabel('vz (μm/μs)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'build/figures/phase_space_{os.path.basename(filename)[:-4]}.pdf')
    plt.close()

def plot_resonance_analysis():
    f_values = [0.1, 0.4, 0.7]
    plt.figure(figsize=(10, 6))
    
    for f in f_values:
        data = np.genfromtxt(f'build/resonance_f{f}00000.csv', delimiter=',', skip_header=1)
        w_V, particles = data.T
        plt.plot(w_V, particles/100, label=f'f={f}')
    
    plt.xlabel('Angular Frequency ωV (MHz)')
    plt.ylabel('Fraction of Particles Remaining')
    plt.title('Resonance Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig('build/figures/resonance_analysis.pdf')
    plt.close()

def main():
    # Create output directory
    os.makedirs('build/figures', exist_ok=True)
    
    # Error analysis
    n_steps = [4000, 8000, 16000, 32000]
    plot_error_analysis(n_steps)
    
    # Phase space analysis
    plot_phase_space('build/two_particles_with_interaction.csv')
    plot_phase_space('build/two_particles_without_interaction.csv')
    
    # Resonance analysis
    plot_resonance_analysis()
    

if __name__ == "__main__":
    main()