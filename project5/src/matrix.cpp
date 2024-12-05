#include "matrix.hpp"
#include <iostream>

Matrix::Matrix(int M, double h, double dt, const arma::cx_mat& V)
    : M_(M), dt_(dt), V_(V) {
    const std::complex<double> I(0, 1);
    r_ = I * dt / (2.0 * h * h);
}

int Matrix::idx(int i, int j) const {
    return i + j*(M_-2);  // Changed indexing to be 0-based
}

void Matrix::build(arma::sp_cx_mat& A, arma::sp_cx_mat& B) {
    int N = (M_-2)*(M_-2);  // Size of matrices
    A.zeros(N, N);
    B.zeros(N, N);
    
    const std::complex<double> I(0, 1);
    
    // Loop through internal points (using 0-based indexing)
    for (int j = 0; j < M_-2; j++) {
        for (int i = 0; i < M_-2; i++) {
            int k = idx(i, j);
            
            // Diagonal elements
            A(k, k) = 1.0 + 4.0*r_ + I*dt_*V_(i+1,j+1)/2.0;
            B(k, k) = 1.0 - 4.0*r_ - I*dt_*V_(i+1,j+1)/2.0;
            
            // x-direction connections
            if (i > 0) {
                A(k, k-1) = -r_;
                B(k, k-1) = r_;
            }
            if (i < M_-3) {
                A(k, k+1) = -r_;
                B(k, k+1) = r_;
            }
            
            // y-direction connections
            if (j > 0) {
                A(k, k-(M_-2)) = -r_;
                B(k, k-(M_-2)) = r_;
            }
            if (j < M_-3) {
                A(k, k+(M_-2)) = -r_;
                B(k, k+(M_-2)) = r_;
            }
        }
    }
}

void Matrix::print_structure(const arma::sp_cx_mat& matrix) {
    std::cout << "Matrix size: " << matrix.n_rows << "x" << matrix.n_cols << std::endl;
    for (int i = 0; i < matrix.n_rows; i++) {
        std::cout << "| ";
        for (int j = 0; j < matrix.n_cols; j++) {
            if (std::abs(matrix(i,j)) > 0)
                std::cout << "• ";
            else
                std::cout << "  ";
        }
        std::cout << "|\n";
    }
    std::cout << std::endl;
}