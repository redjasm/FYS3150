import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Single particle
data = np.genfromtxt('build/single_particle.csv', delimiter=',', skip_header=1)
t, x, y, z = data.T

plt.figure(figsize=(10, 6))
plt.plot(t, z)
plt.xlabel('Time (µs)')
plt.ylabel('z (µm)')
plt.title('Motion in z direction')
plt.savefig('build/figures/single_particle_z.png')
plt.close()

# Two particles
data = np.genfromtxt('build/two_particles.csv', delimiter=',', skip_header=1)
t, x1, y1, z1, x2, y2, z2 = data.T

plt.figure(figsize=(10, 10))
plt.plot(x1, y1, label='Particle 1')
plt.plot(x2, y2, label='Particle 2')
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.title('Motion in xy-plane')
plt.legend()
plt.axis('equal')
plt.savefig('build/figures/two_particles_xy.pdf')
plt.close()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1, y1, z1, label='Particle 1')
ax.plot(x2, y2, z2, label='Particle 2')
ax.set_xlabel('x (µm)')
ax.set_ylabel('y (µm)')
ax.set_zlabel('z (µm)')
ax.set_title('3D trajectory')
plt.legend()
plt.savefig('build/figures/two_particles_3d.pdf')
plt.close()