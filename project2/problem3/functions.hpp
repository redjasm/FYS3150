#include <armadillo>
#include <iostream>
#include <cmath>
#include <fstream>

// 3 a)
double max_offdiag_symmetric(const arma::mat& A, int& k, int& l) {
    int n = A.n_rows;   // Number of rows (and columns) in A
    double max_value = 0.0;

    // Iterate over the upper triangle of the matrix (excluding the diagonal)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double a_ij = std::abs(A(i, j));
            if (a_ij > max_value) {
                max_value = a_ij;
                k = i;
                l = j;
            }
        }
    }
    return max_value;
}