import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create necessary directories
os.makedirs('build/figures', exist_ok=True)

def plot_single_particle(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    t, x, y, z = data.T

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Single Particle Motion')

    axs[0, 0].plot(t, z)
    axs[0, 0].set_xlabel('Time (µs)')
    axs[0, 0].set_ylabel('z (µm)')
    axs[0, 0].set_title('Motion in z direction')

    axs[0, 1].plot(x, y)
    axs[0, 1].set_xlabel('x (µm)')
    axs[0, 1].set_ylabel('y (µm)')
    axs[0, 1].set_title('Motion in xy-plane')
    axs[0, 1].axis('equal')

    axs[1, 0].plot(t, x, label='x')
    axs[1, 0].plot(t, y, label='y')
    axs[1, 0].set_xlabel('Time (µs)')
    axs[1, 0].set_ylabel('Position (µm)')
    axs[1, 0].set_title('x and y vs time')
    axs[1, 0].legend()

    ax3d = fig.add_subplot(224, projection='3d')
    ax3d.plot(x, y, z)
    ax3d.set_xlabel('x (µm)')
    ax3d.set_ylabel('y (µm)')
    ax3d.set_zlabel('z (µm)')
    ax3d.set_title('3D trajectory')

    plt.tight_layout()
    plt.savefig('build/figures/single_particle_motion.pdf')
    plt.close()

def plot_two_particles(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    t, x1, y1, z1, x2, y2, z2 = data.T

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Two Particle Motion')

    axs[0, 0].plot(x1, y1, label='Particle 1')
    axs[0, 0].plot(x2, y2, label='Particle 2')
    axs[0, 0].set_xlabel('x (µm)')
    axs[0, 0].set_ylabel('y (µm)')
    axs[0, 0].set_title('Motion in xy-plane')
    axs[0, 0].legend()
    axs[0, 0].axis('equal')

    axs[0, 1].plot(t, z1, label='Particle 1')
    axs[0, 1].plot(t, z2, label='Particle 2')
    axs[0, 1].set_xlabel('Time (µs)')
    axs[0, 1].set_ylabel('z (µm)')
    axs[0, 1].set_title('Motion in z direction')
    axs[0, 1].legend()

    axs[1, 0].plot(t, np.sqrt(x1**2 + y1**2 + z1**2), label='Particle 1')
    axs[1, 0].plot(t, np.sqrt(x2**2 + y2**2 + z2**2), label='Particle 2')
    axs[1, 0].set_xlabel('Time (µs)')
    axs[1, 0].set_ylabel('Distance from origin (µm)')
    axs[1, 0].set_title('Distance from origin vs time')
    axs[1, 0].legend()

    ax3d = fig.add_subplot(224, projection='3d')
    ax3d.plot(x1, y1, z1, label='Particle 1')
    ax3d.plot(x2, y2, z2, label='Particle 2')
    ax3d.set_xlabel('x (µm)')
    ax3d.set_ylabel('y (µm)')
    ax3d.set_zlabel('z (µm)')
    ax3d.set_title('3D trajectories')
    ax3d.legend()

    plt.tight_layout()
    output_filename = os.path.basename(filename)[:-4]
    plt.savefig(f'build/figures/two_particles_motion_{output_filename}.pdf')
    plt.close()

def plot_random_particles(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    t, particles_inside = data.T

    plt.figure(figsize=(10, 6))
    plt.plot(t, particles_inside)
    plt.xlabel('Time (µs)')
    plt.ylabel('Number of particles inside')
    plt.title('Particles remaining in trap over time')
    plt.savefig('build/figures/random_particles_time_dependent.pdf')
    plt.close()

def main():
    plot_single_particle('build/single_particle.csv')
    plot_two_particles('build/two_particles_with_interaction.csv')
    plot_two_particles('build/two_particles_without_interaction.csv')
    plot_random_particles('build/random_particles_time_dependent.csv')

if __name__ == "__main__":
    main()