#include "matrix.hpp"

int Matrix::to_single_index(int i, int j, int size)
{
    return j * size + i;
}

std::pair<int, int> Matrix::to_pair_index(int k, int size)
{
    int i = k % size;
    int j = k / size;
    return std::make_pair(i, j);
}

arma::sp_cx_mat Matrix::create_tridiagonal(arma::cx_vec &diagonal_values,
                                           std::complex<double> r,
                                           int size,
                                           int block_idx)
{
    arma::sp_cx_mat tri_mat(size, size);

    // Set main diagonal
    for (int i = 0; i < size; i++)
    {
        tri_mat(i, i) = diagonal_values(to_single_index(i, block_idx, size));
    }

    // Set super and sub diagonals
    for (int i = 0; i < size - 1; i++)
    {
        tri_mat(i, i + 1) = r;
        tri_mat(i + 1, i) = r;
    }

    return tri_mat;
}

arma::sp_cx_mat Matrix::create_diagonal(std::complex<double> r, int size)
{
    arma::sp_cx_mat diag_mat(size, size);
    diag_mat.diag().fill(r);
    return diag_mat;
}

arma::sp_cx_mat Matrix::build_matrix(arma::cx_vec &diagonal_values,
                                     std::complex<double> r,
                                     int size)
{
    int total_size = size * size;
    arma::sp_cx_mat matrix(total_size, total_size);

    // Place tridiagonal blocks on main diagonal
    for (int i = 0; i < size; i++)
    {
        int start = i * size;
        int end = start + size - 1;
        matrix.submat(start, start, end, end) =
            create_tridiagonal(diagonal_values, r, size, i);
    }

    // Place diagonal blocks above and below main diagonal
    for (int i = 1; i < size; i++)
    {
        int start = i * size;
        int end = start + size - 1;
        matrix.submat(start - size, start, end - size, end) = create_diagonal(r, size);
        matrix.submat(start, start - size, end, end - size) = create_diagonal(r, size);
    }

    return matrix;
}

std::vector<arma::sp_cx_mat> Matrix::create_AB_matrices(arma::mat &V,
                                                        double h,
                                                        double dt,
                                                        int M)
{
    int size = M - 2;
    std::complex<double> im(0.0, 1.0);
    std::complex<double> r = im * dt / (2.0 * h * h);

    // Initialize vectors for diagonal values
    arma::cx_vec a(size * size);
    arma::cx_vec b(size * size);

    // Fill diagonal values
    for (int j = 0; j < size; j++)
    {
        for (int i = 0; i < size; i++)
        {
            std::complex<double> v_term = im * dt * V(i, j) / 2.0;
            int idx = to_single_index(i, j, size);

            a(idx) = 1.0 + 4.0 * r + v_term;
            b(idx) = 1.0 - 4.0 * r - v_term;
        }
    }

    // Create A and B matrices
    arma::sp_cx_mat A = build_matrix(a, -r, size);
    arma::sp_cx_mat B = build_matrix(b, r, size);

    return {A, B};
}

// Helper function to extract diagonal elements from sparse matrix
arma::cx_vec Matrix::extract_diagonal_elements(const arma::sp_cx_mat &A) {
    arma::cx_vec diagonal(A.n_rows);
    for (size_t i = 0; i < A.n_rows; ++i) {
        diagonal(i) = A(i,i);
    }
    return diagonal;
}

// Gauss-Seidel solver implementation
arma::cx_vec Matrix::solve_gauss_seidel(const arma::sp_cx_mat &A, 
                                       const arma::cx_vec &b,
                                       double tolerance,
                                       int max_iterations) {
    int n = A.n_rows;
    arma::cx_vec x(n, arma::fill::zeros);  // Initial guess x = 0
    arma::cx_vec x_new(n);
    arma::cx_vec diagonal = extract_diagonal_elements(A);

    // Iterate until convergence or max iterations reached
    for (int iter = 0; iter < max_iterations; ++iter) {
        x_new = x;  // Store previous iteration

        // Perform Gauss-Seidel iteration
        for (int i = 0; i < n; ++i) {
            std::complex<double> sum(0.0, 0.0);
            
            // Use iterator to efficiently access non-zero elements
            for (arma::sp_cx_mat::const_row_iterator it = A.begin_row(i); 
                 it != A.end_row(i); ++it) {
                int j = it.col();
                if (j != i) {  // Skip diagonal element
                    if (j < i) {
                        sum += (*it) * x_new(j);  // Use updated values
                    } else {
                        sum += (*it) * x(j);      // Use previous values
                    }
                }
            }
            
            x_new(i) = (b(i) - sum) / diagonal(i);
        }

        // Check convergence
        double relative_error = arma::norm(x_new - x) / arma::norm(x_new);
        if (relative_error < tolerance) {
            return x_new;
        }

        x = x_new;  // Update solution for next iteration
    }

    std::cout << "Warning: Gauss-Seidel method did not converge within " 
              << max_iterations << " iterations" << std::endl;
    return x_new;
}

arma::cx_mat Matrix::create_initial_state(int M, double h,
                                        double x_c, double y_c,
                                        double sigma_x, double sigma_y,
                                        double p_x, double p_y) {
    // Initialize matrix for the entire grid, including boundaries
    arma::cx_mat U(M, M, arma::fill::zeros);
    
    // Complex unit i
    std::complex<double> i(0.0, 1.0);
    
    // Fill the grid with initial state values
    for (int idx = 0; idx < M; idx++) {
        for (int jdx = 0; jdx < M; jdx++) {
            // Calculate x and y coordinates
            double x = idx * h;
            double y = jdx * h;
            
            // Calculate exponential terms
            double gaussian_x = -pow(x - x_c, 2) / (2 * pow(sigma_x, 2));
            double gaussian_y = -pow(y - y_c, 2) / (2 * pow(sigma_y, 2));
            std::complex<double> momentum = i * (p_x * x + p_y * y);
            
            // Set value (ensuring boundary conditions)
            if (idx == 0 || idx == M-1 || jdx == 0 || jdx == M-1) {
                U(idx, jdx) = 0.0;  // Enforce boundary conditions
            } else {
                U(idx, jdx) = exp(gaussian_x + gaussian_y + momentum);
            }
        }
    }
    
    // Normalize the state
    normalize_state(U);
    
    return U;
}

double Matrix::calculate_probability_sum(const arma::cx_mat& U) {
    double sum = 0.0;
    
    // Sum up |u_ij|² for all points
    for (size_t i = 0; i < U.n_rows; ++i) {
        for (size_t j = 0; j < U.n_cols; ++j) {
            sum += std::norm(U(i,j));  // std::norm gives |z|² for complex z
        }
    }
    
    return sum;
}

void Matrix::normalize_state(arma::cx_mat& U) {
    // Calculate current sum of probabilities
    double prob_sum = calculate_probability_sum(U);
    
    // Normalize by dividing each element by sqrt(sum)
    if (prob_sum > 0) {
        U = U / std::sqrt(prob_sum);
    }
}

arma::mat Matrix::create_potential(int M, double h, double v0) {
    arma::mat V(M-2, M-2, arma::fill::zeros);
    
    // Calculate indices for wall position
    int wall_thickness = round(0.02/h);  // Given: wall thickness = 0.02
    int wall_center = round(0.5/h) - 1;  // Wall at x = 0.5
    int y_center = (M-2)/2;              // Center in y-direction
    
    // Calculate slit parameters (in grid points)
    int slit_aperture = round(0.05/h);    // Given: slit aperture = 0.05
    int slit_separation = round(0.05/h);   // Given: separation = 0.05
    
    // Calculate slit center positions (symmetric around y_center)
    int slit1_center = y_center - slit_separation/2;
    int slit2_center = y_center + slit_separation/2;
    
    // Add the wall with slits
    for (int i = wall_center - wall_thickness/2; 
         i <= wall_center + wall_thickness/2; i++) {
        for (int j = 0; j < M-2; j++) {
            // Default: add wall
            V(i,j) = v0;
            
            // Check if we're in a slit region
            bool in_slit1 = (j >= slit1_center - slit_aperture/2) && 
                           (j <= slit1_center + slit_aperture/2);
            bool in_slit2 = (j >= slit2_center - slit_aperture/2) && 
                           (j <= slit2_center + slit_aperture/2);
                
            // If in slit, set potential to 0
            if (in_slit1 || in_slit2) {
                V(i,j) = 0.0;
            }
        }
    }
    
    return V;
}