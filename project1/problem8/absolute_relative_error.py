import numpy as np
import matplotlib.pyplot as plt

# Exact solution for u(x)
# def exact_solution(x):
    # return 1 - (1 - np.exp(-10)) * x - np.exp(-10 * x)

def exact_solution(n):
    data = np.loadtxt("../problem2/solution_data.txt")
    x_exact = data[:, 0]
    u_exact = data[:, 1]
    
    # Interpolate to match n points
    x_resized = np.linspace(0, 1, n)
    u_resized = np.interp(x_resized, x_exact, u_exact)
    
    return x_resized, u_resized


# Function to read data from the C++ output files
def read_data(filename):
    data = np.loadtxt(f"../problem7/{filename}")
    x = data[:, 0]
    v = data[:, 1]
    return x, v

# Compute the absolute and relative error, aligning the data
def error(v_approx, u_exact, x_approx, x_exact):
    # Interpolate the exact solution to the same points as the approximate solution
    # u_interp_func = interp1d(x_exact, u_exact, kind='linear', fill_value="extrapolate")
    # u_exact_interpolated = u_interp_func(x_approx)
    u_exact_interpolated = np.interp(x_approx, x_exact, u_exact)
    
    # Compute absolute and relative errors
    abs_err = np.abs(v_approx - u_exact_interpolated)

    # Avoid divide by zero in relative error calculation
    u_exact_interpolated[u_exact_interpolated == 0] = 1e-16
    
    rel_err = np.abs(abs_err / u_exact_interpolated)
    
    # Avoid log10(0) by setting a small lower limit for errors
    abs_err[abs_err == 0] = 1e-16
    rel_err[rel_err == 0] = 1e-16
    
    log10_abs_err = np.log10(abs_err)
    log10_rel_err = np.log10(rel_err)
    
    return x_approx, log10_abs_err, log10_rel_err

# Function to plot the absolute error for different steps
def plot_absolute_error(filenames, steps):
    plt.figure()

    for n in steps:
        # Get the exact solution with n points
        x_exact, u_exact = exact_solution(n)
        
        for filename in filenames:
            # Read the computed solution data
            x_approx, v_approx = read_data(filename)
            
            # Compute the error
            x_plot, log10_abs_err, _ = error(v_approx, u_exact, x_approx, x_exact)
            
            # Plot the log10 of absolute error
            plt.plot(x_plot, log10_abs_err, label=f"{filename}, n={n}")
    
    plt.xlabel("x")
    plt.ylabel(r"$\log_{10}(\Delta_i)$")
    plt.legend()
    plt.title("Absolute Error for Different Steps")
    plt.savefig("absolute_error_plot.pdf")
    plt.show()

# Function to plot the relative error for different steps
def plot_relative_error(filenames, steps):
    plt.figure()

    for n in steps:
        # Get the exact solution with n points
        x_exact, u_exact = exact_solution(n)
        
        for filename in filenames:
            # Read the computed solution data
            x_approx, v_approx = read_data(filename)
            
            # Compute the error
            x_plot, _, log10_rel_err = error(v_approx, u_exact, x_approx, x_exact)
            
            # Plot the log10 of relative error
            plt.plot(x_plot, log10_rel_err, label=f"{filename}, n={n}")
    
    plt.xlabel("x")
    plt.ylabel(r"$\log_{10}(\epsilon_i)$")
    plt.legend()
    plt.title("Relative Error for Different Steps")
    plt.savefig("relative_error_plot.pdf")
    plt.show()

# List of output files (from different n_steps)
filenames = ["data_1.txt", "data_2.txt", "data_3.txt"]

# List of steps to compare (e.g., 10, 100, 1000)
steps = [10, 100, 1000]

# Plot absolute and relative error for different steps
plot_absolute_error(filenames, steps)
plot_relative_error(filenames, steps)

## Old code, might be useful for reference

# import numpy as np
# import matplotlib.pyplot as plt

# # Function to read data from the C++ output files


# def read_data(filename):
#     data = np.loadtxt(filename)
#     x = data[:, 0]
#     v = data[:, 1]
#     return x, v

# # Load the exact solution data
# def exact_solution():
#     x, v = read_data("../problem2/solution_data.txt")
#     return x, v

# # Function to plot the absolute error
# def plot_absolute_error():
#     plt.figure()

#     for i in range(1, 7):
#         x, v = read_data(f"../problem7/data_{i}.txt")
#         plt.plot(x, np.log10(np.abs(exact_solution(x) - v)),
#                  label=f"Solution {i}")
        
#     # Load the exact solution data
#     u = exact_solution()

#     delta = np.abs(u - v)

#     # Avoid log(0) by adding a small value to delta
#     delta += 1e-15

#     plt.plot(x, np.log10(delta), label="Exact Solution", color='blue')


#     plt.xlabel("x")
#     plt.ylabel(r"$\log_{10}(\Delta_i)$")
#     plt.legend()
#     plt.title("Absolute Error")
#     plt.show()


# plot_absolute_error()