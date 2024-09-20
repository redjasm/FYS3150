#include "../problem4/functions.hpp"

int main() {
    // Parameters
    int n = 10;
    int N = n - 1;
    double h = 1.0 / n;
    double a = -1.0 / (h * h);
    double d = 2.0 / (h * h);

    // Construct matrix A
    arma::mat A = arma::mat(N, N, arma::fill::zeros);
    A.diag().fill(d);
    A.diag(-1).fill(a);
    A.diag(1).fill(a);

    // Solve eigenvalue problem
    arma::vec eigenvalues;
    arma::mat eigenvectors;
    int iterations = 0;
    double eps = 1e-8;
    int max_iterations = 10000000;

    jacobi_eigensolver(A, eigenvalues, eigenvectors, iterations, eps, max_iterations);

    // Sort eigenvalues and eigenvectors
    arma::uvec indices = arma::sort_index(eigenvalues);
    eigenvalues = eigenvalues.elem(indices);
    eigenvectors = eigenvectors.cols(indices);

    // Extract the three lowest eigenvectors
    arma::mat eigenvectors_lowest = eigenvectors.cols(0, 2);

    // Extend eigenvectors with boundary conditions
    int total_points = n + 1;
    arma::vec x = arma::linspace(0.0, 1.0, total_points);

    arma::mat eigenvectors_extended = arma::mat(total_points, 3, arma::fill::zeros);

    for (int i = 0; i < 3; ++i) {
        eigenvectors_extended.col(i).subvec(1, N) = eigenvectors_lowest.col(i);
    }

    // Compute analytical eigenvectors
    arma::mat eigvec_analytical = arma::mat(N, N);

    for (int j = 1; j <= N; ++j) {
        for (int i = 1; i <= N; ++i) {
            eigvec_analytical(i - 1, j - 1) = sin(i * j * M_PI / (N + 1));
        }
        eigvec_analytical.col(j - 1) = arma::normalise(eigvec_analytical.col(j - 1));
    }

    // Extend analytical eigenvectors
    arma::mat eigvec_analytical_extended = arma::mat(total_points, 3, arma::fill::zeros);

    for (int i = 0; i < 3; ++i) {
        eigvec_analytical_extended.col(i).subvec(1, N) = eigvec_analytical.col(i);
    }

    // Write data to files
    std::ofstream file_num("eigenvectors_numerical_n10.txt");
    std::ofstream file_ana("eigenvectors_analytical_n10.txt");

    for (int i = 0; i < total_points; ++i) {
        file_num << x(i);
        file_ana << x(i);

        for (int j = 0; j < 3; ++j) {
            file_num << " " << eigenvectors_extended(i, j);
            file_ana << " " << eigvec_analytical_extended(i, j);
        }
        file_num << "\n";
        file_ana << "\n";
    }

    file_num.close();
    file_ana.close();

    return 0;
}
