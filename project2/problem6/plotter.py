import numpy as np
import matplotlib.pyplot as plt

# Load numerical eigenvectors
data_num = np.loadtxt('eigenvectors_numerical_n10.txt')
x = data_num[:, 0]
v1_num = data_num[:, 1]
v2_num = data_num[:, 2]
v3_num = data_num[:, 3]

# Load analytical eigenvectors
data_ana = np.loadtxt('eigenvectors_analytical_n10.txt')
v1_ana = data_ana[:, 1]
v2_ana = data_ana[:, 2]
v3_ana = data_ana[:, 3]

# Plot the first eigenvector
plt.figure(figsize=(10, 6))
plt.plot(x, v1_num, 'bo-', label='Numerical Eigenvector 1')
plt.plot(x, v1_ana, 'r--', label='Analytical Eigenvector 1')
plt.title('Eigenvector 1 (n=10)')
plt.xlabel('x')
plt.ylabel('v')
plt.legend()
plt.grid(True)
plt.savefig('eigenvector1_n10.pdf')
plt.show()

# Plot the second eigenvector
plt.figure(figsize=(10, 6))
plt.plot(x, v2_num, 'bo-', label='Numerical Eigenvector 2')
plt.plot(x, v2_ana, 'r--', label='Analytical Eigenvector 2')
plt.title('Eigenvector 2 (n=10)')
plt.xlabel('x')
plt.ylabel('v')
plt.legend()
plt.grid(True)
plt.savefig('eigenvector2_n10.pdf')
plt.show()

# Plot the third eigenvector
plt.figure(figsize=(10, 6))
plt.plot(x, v3_num, 'bo-', label='Numerical Eigenvector 3')
plt.plot(x, v3_ana, 'r--', label='Analytical Eigenvector 3')
plt.title('Eigenvector 3 (n=10)')
plt.xlabel('x')
plt.ylabel('v')
plt.legend()
plt.grid(True)
plt.savefig('eigenvector3_n10.pdf')
plt.show()
