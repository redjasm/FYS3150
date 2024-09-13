import numpy as np
import matplotlib.pyplot as plt

# Load data from the file
def load_data(filename):
    data = np.loadtxt(filename)
    x = data[:, 0]
    v = data[:, 1]
    return x, v

plt.figure(figsize=(10, 6))

for i in range(1, 7):
    x, v = load_data(f"data_{i}.txt")
    plt.plot(x, v, label=f"Solution {i}")

# Load the exact solution data
x, v = load_data("../problem2/solution_data.txt")
plt.plot(x, v, label="Exact Solution", color='blue')

# Adding plot labels and title
plt.xlabel("x", fontsize=12)
plt.ylabel("u(x)", fontsize=12)
plt.title("Comparison of Numerical and Exact Solutions for Different Discretizations", fontsize=14)
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig("general_algorithm_plot.pdf")

# Show the plot
plt.show()