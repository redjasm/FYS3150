#include "matrix.hpp"

int Matrix::to_single_index(int i, int j, int size)
{
    return j * size + i;
}

std::pair<int, int> Matrix::to_pair_index(int k, int size)
{
    int i = k % size;
    int j = k / size;
    return std::make_pair(i, j);
}

arma::sp_cx_mat Matrix::create_tridiagonal(arma::cx_vec &diagonal_values,
                                           std::complex<double> r,
                                           int size,
                                           int block_idx)
{
    arma::sp_cx_mat tri_mat(size, size);

    // Set main diagonal
    for (int i = 0; i < size; i++)
    {
        tri_mat(i, i) = diagonal_values(to_single_index(i, block_idx, size));
    }

    // Set super and sub diagonals
    for (int i = 0; i < size - 1; i++)
    {
        tri_mat(i, i + 1) = r;
        tri_mat(i + 1, i) = r;
    }

    return tri_mat;
}

arma::sp_cx_mat Matrix::create_diagonal(std::complex<double> r, int size)
{
    arma::sp_cx_mat diag_mat(size, size);
    diag_mat.diag().fill(r);
    return diag_mat;
}

arma::sp_cx_mat Matrix::build_matrix(arma::cx_vec &diagonal_values,
                                     std::complex<double> r,
                                     int size)
{
    int total_size = size * size;
    arma::sp_cx_mat matrix(total_size, total_size);

    // Place tridiagonal blocks on main diagonal
    for (int i = 0; i < size; i++)
    {
        int start = i * size;
        int end = start + size - 1;
        matrix.submat(start, start, end, end) =
            create_tridiagonal(diagonal_values, r, size, i);
    }

    // Place diagonal blocks above and below main diagonal
    for (int i = 1; i < size; i++)
    {
        int start = i * size;
        int end = start + size - 1;
        matrix.submat(start - size, start, end - size, end) = create_diagonal(r, size);
        matrix.submat(start, start - size, end, end - size) = create_diagonal(r, size);
    }

    return matrix;
}

std::vector<arma::sp_cx_mat> Matrix::create_AB_matrices(arma::mat &V,
                                                        double h,
                                                        double dt,
                                                        int M)
{
    int size = M - 2;
    std::complex<double> im(0.0, 1.0);
    std::complex<double> r = im * dt / (2.0 * h * h);

    // Initialize vectors for diagonal values
    arma::cx_vec a(size * size);
    arma::cx_vec b(size * size);

    // Fill diagonal values
    for (int j = 0; j < size; j++)
    {
        for (int i = 0; i < size; i++)
        {
            std::complex<double> v_term = im * dt * V(i, j) / 2.0;
            int idx = to_single_index(i, j, size);

            a(idx) = 1.0 + 4.0 * r + v_term;
            b(idx) = 1.0 - 4.0 * r - v_term;
        }
    }

    // Create A and B matrices
    arma::sp_cx_mat A = build_matrix(a, -r, size);
    arma::sp_cx_mat B = build_matrix(b, r, size);

    return {A, B};
}