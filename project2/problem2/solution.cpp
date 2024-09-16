#include <armadillo>
#include <iostream>
#include <cmath>

int main() {
    int N = 6;                 // Matrix size
    int n = N + 1;             // Number of discretization points
    double h = 1.0 / n;        // Step size
    double a = -1.0 / (h * h); // Off-diagonal elements
    double d = 2.0 / (h * h);  // Diagonal elements

    arma::mat A = arma::mat(N, N, arma::fill::zeros);

    // Set up tridiagonal matrix A
    A.diag(0).fill(d);         // Main diagonal
    A.diag(-1).fill(a);        // Lower diagonal
    A.diag(1).fill(a);         // Upper diagonal

    // Compute eigenvalues and eigenvectors using Armadillo
    arma::vec eigval;
    arma::mat eigvec;

    arma::eig_sym(eigval, eigvec, A);

    // Normalize eigenvectors
    eigvec = arma::normalise(eigvec);

    // Compute analytical eigenvalues and eigenvectors
    arma::vec eigval_analytical(N);
    arma::mat eigvec_analytical(N, N);

    for (int j = 1; j <= N; ++j) {
        eigval_analytical(j - 1) = d + 2 * a * cos(j * M_PI / (N + 1));

        for (int i = 1; i <= N; ++i) {
            eigvec_analytical(i - 1, j - 1) = sin(i * j * M_PI / (N + 1));
        }
    }

    // Normalize analytical eigenvectors
    eigvec_analytical = arma::normalise(eigvec_analytical);

    // Compare eigenvalues
    std::cout << "Comparing Eigenvalues:\n";
    for (int i = 0; i < N; ++i) {
        double diff = std::abs(eigval(i) - eigval_analytical(i));
        std::cout << "Eigenvalue " << i + 1 << " difference: " << diff << "\n";
    }

    // Compare eigenvectors
    std::cout << "\nComparing Eigenvectors:\n";
    for (int j = 0; j < N; ++j) {
        // Account for possible sign differences
        arma::vec vec_diff = arma::abs(eigvec.col(j)) - arma::abs(eigvec_analytical.col(j));
        double max_diff = arma::max(arma::abs(vec_diff));
        std::cout << "Eigenvector " << j + 1 << " max difference: " << max_diff << "\n";
    }

    return 0;
}
