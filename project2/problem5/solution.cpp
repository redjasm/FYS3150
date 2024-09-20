#include "../problem4/functions.hpp"
#include <fstream>

int main()
{
    std::ofstream data_file("jacobi_iterations.txt");
    data_file << "N\tIterations\n";

    for (int N = 10; N <= 320; N *= 2) // 640 takes a long time, maybe use 320 instead? (Food for thought)
    {
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
        int max_iterations = 10000000;
        int iterations = 0;

        // Vectors to store eigenvalues and eigenvectors
        arma::vec eigenvalues;
        arma::mat eigenvectors;

        // Call Jacobi's method
        jacobi_eigensolver(A, eigenvalues, eigenvectors, iterations, eps, max_iterations);

        // Output the results
        std::cout << "N = " << N << ", Iterations = " << iterations << "\n";
        data_file << N << "\t" << iterations << "\n";
    }

    data_file.close();
    return 0;
}