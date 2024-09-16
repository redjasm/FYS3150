import numpy as np
import matplotlib.pyplot as plt

def exact_solution(x):
    return 1 - (1 - np.exp(-10)) * x - np.exp(-10 * x)

def read_data(filename):
    data = np.loadtxt(f"../problem7/{filename}")
    x = data[:, 0]
    v = data[:, 1]
    return x, v

def compute_errors(x, v):
    # Exclude boundary points
    x_interior = x[1:-1]
    v_interior = v[1:-1]
    u_interior = exact_solution(x_interior)
    
    # Compute absolute error
    abs_err = np.abs(u_interior - v_interior)
    
    # Compute relative error, avoiding division by zero
    rel_err = abs_err / np.abs(u_interior)
    
    # Replace zeros with a small number to avoid log(0)
    abs_err[abs_err == 0] = 1e-16
    rel_err[rel_err == 0] = 1e-16
    
    # Compute logarithms
    log_abs_err = np.log10(abs_err)
    log_rel_err = np.log10(rel_err)
    
    return x_interior, log_abs_err, log_rel_err

def plot_absolute_error(filenames, steps):
    plt.figure()
    for filename, n in zip(filenames, steps):
        x, v = read_data(filename)
        x_plot, log_abs_err, _ = compute_errors(x, v)
        plt.plot(x_plot, log_abs_err, label=f"n={n}")
    plt.xlabel("x")
    plt.ylabel(r"$\log_{10}(|u_i - v_i|)$")
    plt.title("Absolute Error")
    plt.legend()
    plt.grid(True)
    plt.savefig("absolute_error_plot.pdf")
    plt.show()

def plot_relative_error(filenames, steps):
    plt.figure()
    for filename, n in zip(filenames, steps):
        x, v = read_data(filename)
        x_plot, _, log_rel_err = compute_errors(x, v)
        plt.plot(x_plot, log_rel_err, label=f"n={n}")
    plt.xlabel("x")
    plt.ylabel(r"$\log_{10}\left(\left|\dfrac{u_i - v_i}{u_i}\right|\right)$")
    plt.title("Relative Error")
    plt.legend()
    plt.grid(True)
    plt.savefig("relative_error_plot.pdf")
    plt.show()

def compute_max_relative_errors(filenames, steps):
    max_rel_errors = []
    for filename in filenames:
        x, v = read_data(filename)
        _, _, log_rel_err = compute_errors(x, v)
        rel_err = 10 ** log_rel_err
        max_rel_errors.append(np.max(rel_err))
    return steps, max_rel_errors

def plot_max_relative_errors(steps, max_rel_errors):
    plt.figure()
    plt.loglog(steps, max_rel_errors, marker='o')
    plt.xlabel(r"$n_{\mathrm{steps}}$")
    plt.ylabel(r"$\max\left(\epsilon_i\right)$")
    plt.title("Maximum Relative Error vs $n_{\mathrm{steps}}$")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig("max_relative_error_plot.pdf")
    plt.show()

# Main execution
filenames = ["data_1.txt", "data_2.txt", "data_3.txt"]
steps = [10, 100, 1000]

# Plot errors
plot_absolute_error(filenames, steps)
plot_relative_error(filenames, steps)

# Compute and plot maximum relative errors
steps, max_rel_errors = compute_max_relative_errors(filenames, steps)
print("n_steps\tMax Relative Error")
for n, err in zip(steps, max_rel_errors):
    print(f"{n}\t{err}")
plot_max_relative_errors(steps, max_rel_errors)
