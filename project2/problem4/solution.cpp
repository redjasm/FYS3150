#include "functions.hpp"

// Functions from Problem 4a and 3a
double max_offdiag_symmetric(const arma::mat& A, int& k, int& l);
void jacobi_rotate(arma::mat& A, arma::mat& R, int k, int l);
void jacobi_eigensolver(arma::mat& A, arma::vec& eigenvalues, arma::mat& eigenvectors, int& iterations, double eps, int max_iterations);

int main() {
    int N = 6;
    double n_steps = N + 1;
    double h = 1.0 / n_steps;
    double a = -1.0 / (h * h);
    double d = 2.0 / (h * h);

    // Construct tridiagonal matrix A
    arma::mat A = arma::mat(N, N, arma::fill::zeros);
    A.diag().fill(d);
    A.diag(-1).fill(a);
    A.diag(1).fill(a);

    // Parameters for Jacobi's method
    double eps = 1e-8;
    int max_iterations = 1000000; 
    int iterations = 0;

    // Vectors to store eigenvalues and eigenvectors
    arma::vec eigenvalues;
    arma::mat eigenvectors;

    // Call Jacobi's method
    jacobi_eigensolver(A, eigenvalues, eigenvectors, iterations, eps, max_iterations);

    // Sort numerical eigenvalues and eigenvectors
    arma::uvec indices = arma::sort_index(eigenvalues);
    eigenvalues = eigenvalues.elem(indices);
    eigenvectors = eigenvectors.cols(indices);

    // Compute analytical eigenvalues and eigenvectors
    arma::vec eigval_analytical(N);
    arma::mat eigvec_analytical(N, N);

    for (int j = 1; j <= N; ++j) {
        eigval_analytical(j - 1) = d + 2 * a * cos(j * M_PI / (N + 1));
        for (int i = 1; i <= N; ++i) {
            eigvec_analytical(i - 1, j - 1) = sin(i * j * M_PI / (N + 1));
        }
    }

    // Normalize the analytical eigenvectors
    eigvec_analytical = arma::normalise(eigvec_analytical);

    // Sort analytical eigenvalues and eigenvectors
    arma::uvec indices_analytical = arma::sort_index(eigval_analytical);
    eigval_analytical = eigval_analytical.elem(indices_analytical);
    eigvec_analytical = eigvec_analytical.cols(indices_analytical);

    // Compare eigenvalues
    arma::vec eigval_diff = arma::abs(eigenvalues - eigval_analytical);

    // Output the differences
    std::cout << "Eigenvalues from Jacobi's method:\n" << eigenvalues << "\n";
    std::cout << "Analytical eigenvalues:\n" << eigval_analytical << "\n";
    std::cout << "Difference in eigenvalues:\n" << eigval_diff << "\n";

    // Compare eigenvectors
    std::cout << "\nComparing Eigenvectors:\n";
    for (int j = 0; j < N; ++j) {
        arma::vec vec_diff = arma::abs(arma::abs(eigenvectors.col(j)) - arma::abs(eigvec_analytical.col(j)));
        double max_diff = arma::max(vec_diff);
        std::cout << "Eigenvector " << j + 1 << " max difference: " << max_diff << "\n";
    }

    std::cout << "\nNumber of iterations: " << iterations << "\n";

    return 0;
}