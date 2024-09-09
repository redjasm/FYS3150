import numpy as np
import matplotlib.pyplot as plt

# Load data from the file
data = np.loadtxt("solution_data.txt")

# Split data into x and u(x)
x_values = data[:, 0]
u_values = data[:, 1]

# Create the plot
plt.plot(x_values, u_values, label="Exact Solution", color='blue')

# Add labels and title
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Exact Solution of the Poisson Equation")

# Show grid
plt.grid(True)

# Add a legend
plt.legend()

# Save the plot to a file
plt.savefig("poisson_solution_plot.png")

# Show the plot
plt.show()
