#include <armadillo>
#include <iostream>
#include <cmath>

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

// Function declaration
double max_offdiag_symmetric(const arma::mat& A, int& k, int& l);

// 3 b) Test the function
int main() {
    // Define the symmetric matrix A
    arma::mat A = arma::mat(4, 4).fill(0.0);
    A(0, 0) = 1.0;    A(0, 1) = 0.0;    A(0, 2) = 0.0;    A(0, 3) = 0.5;
    A(1, 0) = 0.0;    A(1, 1) = 1.0;    A(1, 2) = -0.7;   A(1, 3) = 0.0;
    A(2, 0) = 0.0;    A(2, 1) = -0.7;   A(2, 2) = 1.0;    A(2, 3) = 0.0;
    A(3, 0) = 0.5;    A(3, 1) = 0.0;    A(3, 2) = 0.0;    A(3, 3) = 1.0;

    // Variables to hold the indices of the largest off-diagonal element
    int k = 0;
    int l = 0;

    // Call the function
    double max_value = max_offdiag_symmetric(A, k, l);

    // Output the results
    std::cout << "The largest off-diagonal element is A(" << k << "," << l << ") = " << A(k, l) << std::endl;
    std::cout << "Its absolute value is " << max_value << std::endl;

    return 0;
}