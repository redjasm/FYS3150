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

    // Helper function for Gauss-Seidel solver
    arma::cx_vec extract_diagonal_elements(const arma::sp_cx_mat &A);

    // Helper function for normalization
    double calculate_probability_sum(const arma::cx_mat& U);

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

    // Gauss-Seidel solver
    arma::cx_vec solve_gauss_seidel(const arma::sp_cx_mat &A, 
                                   const arma::cx_vec &b,
                                   double tolerance = 1e-10,
                                   int max_iterations = 1000);

    // Set up initial state
    arma::cx_mat create_initial_state(int M, double h, 
                                    double x_c, double y_c,
                                    double sigma_x, double sigma_y,
                                    double p_x, double p_y);

    // Normalize state
    void normalize_state(arma::cx_mat& U);

    // Create potential for double-slit setup
    arma::mat create_potential(int M, double h, double v0, int n_slits);
};
#endif