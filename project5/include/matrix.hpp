#ifndef MATRIX_H
#define MATRIX_H

#include <armadillo>
#include <complex>
#include <vector>

class Matrix
{
private:
    // Helper functions for creating matrix structure
    arma::sp_cx_mat create_tridiagonal(arma::cx_vec &diagonal_values,
                                       std::complex<double> r,
                                       int size,
                                       int block_idx);

    arma::sp_cx_mat create_diagonal(std::complex<double> r, int size);

public:
    // Default constructor
    Matrix() = default;

    // Main function to create A and B matrices
    std::vector<arma::sp_cx_mat> create_AB_matrices(arma::mat &V,
                                                    double h,
                                                    double dt,
                                                    int M);

    // Index conversion utilities
    static int to_single_index(int i, int j, int size);
    static std::pair<int, int> to_pair_index(int k, int size);

    // Create full matrix with given block structure
    arma::sp_cx_mat build_matrix(arma::cx_vec &diagonal_values,
                                 std::complex<double> r,
                                 int size);
};

#endif