# Project 5: Schrödinger Equation Solver

This project implements a numerical solver for the 2D time-dependent Schrödinger equation, focusing on simulating quantum wave packet dynamics in a double-slit setup.

## Project Structure

```
project5/
├── build/
├── include/
│   ├── matrix.hpp
│   └── simulation.hpp
└── src/
    ├── matrix.cpp
    ├── simulation.cpp
    └── main.cpp

```

## Problem 1: Deriving the Crank-Nicolson Scheme

### Mathematical Derivation

Starting from the 2D time-dependent Schrödinger equation:

```
i∂u/∂t = -∂²u/∂x² - ∂²u/∂y² + v(x,y)u

```

We derived the Crank-Nicolson discretization:

```
u_{ij}^{n+1} - r[u_{i+1,j}^{n+1} - 2u_{ij}^{n+1} + u_{i-1,j}^{n+1}]
- r[u_{i,j+1}^{n+1} - 2u_{ij}^{n+1} + u_{i,j-1}^{n+1}]
+ (iΔt/2)v_{ij}u_{ij}^{n+1} =
u_{ij}^n + r[u_{i+1,j}^n - 2u_{ij}^n + u_{i-1,j}^n]
+ r[u_{i,j+1}^n - 2u_{ij}^n + u_{i,j-1}^n]
- (iΔt/2)v_{ij}u_{ij}^n

```

where r = iΔt/(2h²)

## Problem 2: Matrix Implementation

### Key Implementations

1. Index conversion utilities in `matrix.hpp`:

```cpp
static int to_single_index(int i, int j, int size);
static std::pair<int, int> to_pair_index(int k, int size);

```

1. Matrix construction functions:

```cpp
arma::sp_cx_mat create_tridiagonal(arma::cx_vec &diagonal_values,
                                   std::complex<double> r,
                                   int size,
                                   int block_idx);
arma::sp_cx_mat create_diagonal(std::complex<double> r, int size);
arma::sp_cx_mat build_matrix(arma::cx_vec &diagonal_values,
                             std::complex<double> r,
                             int size);

```

### Test Results

Tested with M=5 (3x3 internal grid):

```
Matrix A structure:
| • •   •           |
| • • •   •         |
|   • •     •       |
| •     • •   •     |
|   •   • • •   •   |
|     •   • •     • |
|       •     • •   |
|         •   • • • |
|           •   • • |

```

Matrix size: 9x9, Non-zero elements: 33

Similar successful test with M=6 (4x4 internal grid) showing correct structure scaling.

## Problem 3: Gauss-Seidel Solver

### Implementation

Added to `matrix.hpp`:

```cpp
arma::cx_vec solve_gauss_seidel(const arma::sp_cx_mat &A,
                               const arma::cx_vec &b,
                               double tolerance = 1e-10,
                               int max_iterations = 1000);

```

### Results

- Relative residual: 2.830e-15
- Solution vector shows expected symmetry:
    
    ```
    x[0] ≈ x[2] ≈ x[6] ≈ x[8] ≈ 1.0 - 0.016i  (corners)
    x[1] ≈ x[3] ≈ x[5] ≈ x[7] ≈ 1.0 - 0.008i  (edges)
    x[4] ≈ 1.0 - 8e-6i                        (center)
    
    ```
    

## Problem 4: Initial State Implementation

### Key Functions

Added to `matrix.hpp`:

```cpp
arma::cx_mat create_initial_state(int M, double h,
                                double x_c, double y_c,
                                double sigma_x, double sigma_y,
                                double p_x, double p_y);
void normalize_state(arma::cx_mat& U);
double calculate_probability_sum(const arma::cx_mat& U);

```

### Test Results

With parameters:

- M = 201 (200 steps)
- h = 1/200
- x_c = 0.25, y_c = 0.5
- σ_x = 0.05, σ_y = 0.05
- p_x = 200, p_y = 0

Results:

- Total probability: 1.000e+00 (perfect normalization)
- U(100,100) ≈ 1.813e-07 - 1.065e-07i (far from center)
- U(50,100) ≈ 5.444e-02 - 1.480e-02i (at center)
- U(0,0) = 0 (boundary condition satisfied)

## Problem 5: Double-Slit Potential

### Implementation

Added to `matrix.hpp`:

```cpp
arma::mat create_potential(int M, double h, double v0);

```

### Test Results

With parameters:

- M = 201
- h = 1/200
- v0 = 1e10

Results show correct:

- Wall thickness (0.02)
- Wall position (x = 0.5)
- Slit aperture (0.05)
- Slit separation (0.05)
- Symmetry around y = 0.5

Potential values:

```
Wall center value: 1.000e+10
Slit 1 value: 0.000e+00
Slit 2 value: 0.000e+00

```

Visual representation confirms correct double-slit structure:

```
•••••           •••••|••••           •••••

```

## Problem 6: Full Simulation Implementation

### Implementation Details

1. **Parameter Handling**
    - Implemented simulation parameters as constants
    - Added command-line configuration capabilities
    - Created output directory structure in build folder
2. **Core Components**
    - Potential matrix setup
    - Initial state initialization
    - Crank-Nicolson matrices creation
    - Time evolution loop
    - Data storage using `arma::cx_cube`
3. **Data Storage**
    - Used `arma::cx_cube` for efficient storage of time evolution
    - Dimensions: (M-2) × (M-2) × (n_timesteps + 1)
    - Stores complex wave function values at each time step
4. **Progress Tracking**
    - Added progress bar with percentage completion
    - Included time estimation and runtime statistics
    - Console output for simulation milestones

## Problem 7: Probability Conservation (In Progress)

### Implementation

1. **Test Cases Setup**
    - No barrier case:
        
        ```
        h = 0.005
        Δt = 2.5×10⁻⁵
        T = 0.008
        x_c = 0.25
        σ_x = 0.05
        p_x = 200
        y_c = 0.5
        σ_y = 0.05
        p_y = 0
        v0 = 0
        
        ```
        
    - With barrier case:
        
        ```
        Same parameters but:
        v0 = 1×10¹⁰
        σ_y = 0.10
        
        ```
        
2. **Progress**
    - Implemented both test cases
    - Added probability deviation tracking
    - Currently testing simulation runtime performance

###