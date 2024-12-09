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