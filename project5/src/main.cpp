#include "matrix.hpp"
#include <iostream>

int main() {
    // Test case with 5x5 grid (3x3 internal points)
    int M = 5;
    double h = 0.1;
    double dt = 0.01;
    
    // Create test potential matrix (zeros)
    arma::cx_mat V(M, M, arma::fill::zeros);
    
    // Create matrices
    Matrix matrix(M, h, dt, V);
    arma::sp_cx_mat A, B;
    
    // Build and print matrices
    matrix.build(A, B);
    
    std::cout << "Matrix A:" << std::endl;
    Matrix::print_structure(A);
    
    std::cout << "Matrix B:" << std::endl;
    Matrix::print_structure(B);
    
    // Print some validation info
    std::cout << "Number of non-zero elements in A: " << A.n_nonzero << std::endl;
    std::cout << "Number of non-zero elements in B: " << B.n_nonzero << std::endl;
    
    return 0;
}