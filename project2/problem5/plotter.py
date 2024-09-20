import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file
data = np.loadtxt('jacobi_iterations.txt', skiprows=1)  # Skip the header line

# Separate the data into N and Iterations
N = data[:, 0]
iterations = data[:, 1]

# Create a linear plot
plt.figure(figsize=(10, 6))
plt.plot(N, iterations, marker='o', linestyle='-', color='b')
plt.title('Number of Jacobi Iterations vs. Matrix Size N (Linear Scale)')
plt.xlabel('Matrix Size N')
plt.ylabel('Number of Iterations')
plt.grid(True)
plt.savefig('jacobi_iterations_linear.pdf')
plt.show()

# Create a log-log plot
plt.figure(figsize=(10, 6))
plt.loglog(N, iterations, marker='o', linestyle='-', color='r')
plt.title('Number of Jacobi Iterations vs. Matrix Size N (Log-Log Scale)')
plt.xlabel('Matrix Size N (log scale)')
plt.ylabel('Number of Iterations (log scale)')
plt.grid(True, which="both", ls="--")
plt.savefig('jacobi_iterations_loglog.pdf')
plt.show()

# Fit a line to the log-log data to determine scaling exponent
log_N = np.log2(N)
log_iterations = np.log2(iterations)
coefficients = np.polyfit(log_N, log_iterations, 1)
scaling_exponent = coefficients[0]
print(f"Estimated scaling exponent: {scaling_exponent:.2f}")

# Plot the fitted line on the log-log plot
plt.figure(figsize=(10, 6))
plt.loglog(N, iterations, 'o', label='Data')
plt.loglog(N, 2**(coefficients[1]) * N**scaling_exponent, '-', label=f'Fit: iterations ∝ N^{scaling_exponent:.2f}')
plt.title('Number of Jacobi Iterations vs. Matrix Size N (Log-Log Scale with Fit)')
plt.xlabel('Matrix Size N (log scale)')
plt.ylabel('Number of Iterations (log scale)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('jacobi_iterations_loglog_fit.pdf')
plt.show()
