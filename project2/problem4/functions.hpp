// Include the function and headers from problem3
#include "../problem3/functions.hpp"

/**
 * @brief Performs a single Jacobi rotation to eliminate the off-diagonal element A(k, l).
 *
 * This function modifies the input matrix A by performing a Jacobi rotation that zeros out
 * the element A(k, l) (and A(l, k) due to symmetry). It also updates the eigenvector matrix R
 * to keep track of the accumulated rotations.
 *
 * @param[in,out] A Symmetric matrix to be diagonalized. On input, the original matrix A.
 *                  On output, the updated matrix after the Jacobi rotation.
 * @param[in,out] R Matrix whose columns are the eigenvectors. On input, typically the identity matrix.
 *                  On output, updated with the rotation applied.
 * @param[in]     k Row index of the largest off-diagonal element.
 * @param[in]     l Column index of the largest off-diagonal element.
 */
void jacobi_rotate(arma::mat &A, arma::mat &R, int k, int l)
{
    int n = A.n_rows;
    double tau, t, c, s;

    if (A(k, l) != 0.0)
    {
        tau = (A(l, l) - A(k, k)) / (2.0 * A(k, l));

        if (tau >= 0)
        {
            t = 1.0 / (tau + std::sqrt(1.0 + tau * tau));
        }
        else
        {
            t = -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
        }

        c = 1.0 / std::sqrt(1.0 + t * t);
        s = t * c;
    }
    else
    {
        c = 1.0;
        s = 0.0;
    }

    double a_kk = A(k, k);
    double a_ll = A(l, l);
    double a_kl = A(k, l);

    // Update matrix elements
    A(k, k) = c * c * a_kk - 2.0 * c * s * a_kl + s * s * a_ll;
    A(l, l) = s * s * a_kk + 2.0 * c * s * a_kl + c * c * a_ll;
    A(k, l) = 0.0; // Off-diagonal elements set to zero
    A(l, k) = 0.0;

    for (int i = 0; i < n; ++i)
    {
        if (i != k && i != l)
        {
            double a_ik = A(i, k);
            double a_il = A(i, l);

            A(i, k) = c * a_ik - s * a_il;
            A(k, i) = A(i, k);

            A(i, l) = c * a_il + s * a_ik;
            A(l, i) = A(i, l);
        }

        // Update the eigenvector matrix
        double r_ik = R(i, k);
        double r_il = R(i, l);

        R(i, k) = c * r_ik - s * r_il;
        R(i, l) = c * r_il + s * r_ik;
    }
}

/**
 * @brief Computes the eigenvalues and eigenvectors of a symmetric matrix using Jacobi's method.
 *
 * This function iteratively applies Jacobi rotations to the input matrix A until all off-diagonal
 * elements are below a specified tolerance. The diagonal elements of the resulting matrix A
 * are the eigenvalues, and the columns of the matrix R are the corresponding normalized eigenvectors.
 *
 * @param[in,out] A             Symmetric matrix to diagonalize. On output, contains the diagonalized matrix.
 * @param[out]    eigenvalues   Vector to store the computed eigenvalues.
 * @param[out]    eigenvectors  Matrix to store the computed eigenvectors as columns.
 * @param[out]    iterations    Number of iterations performed.
 * @param[in]     eps           Convergence tolerance for the off-diagonal elements.
 * @param[in]     max_iterations Maximum number of iterations allowed.
 */

void jacobi_eigensolver(arma::mat& A, arma::vec& eigenvalues, arma::mat& eigenvectors, int& iterations, double eps, int max_iterations) {
    int n = A.n_rows;
    arma::mat R = arma::mat(n, n, arma::fill::eye); // Initialize R as identity matrix
    iterations = 0;

    int k, l;
    double max_offdiag = max_offdiag_symmetric(A, k, l);

    while (max_offdiag > eps && iterations < max_iterations) {
        jacobi_rotate(A, R, k, l);
        max_offdiag = max_offdiag_symmetric(A, k, l);
        iterations++;
    }

    eigenvalues = A.diag();
    eigenvectors = R;
}