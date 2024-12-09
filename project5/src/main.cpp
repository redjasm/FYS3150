// test_matrix.cpp
#include "matrix.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

// A function that prints the structure of a sparse matrix to screen.
void print_sp_matrix_structure(const arma::sp_cx_mat &A)
{
    using namespace std;
    using namespace arma;

    // Declare a C-style 2D array of strings.
    string S[A.n_rows][A.n_cols];

    // Initialise all the strings to " ".
    for (int i = 0; i < A.n_rows; i++)
    {
        for (int j = 0; j < A.n_cols; j++)
        {
            S[i][j] = " ";
        }
    }

    // Next, we want to set the string to a dot at each non-zero element.
    // To do this we use the special loop iterator from the sp_cx_mat class
    // to help us loop over only the non-zero matrix elements.
    sp_cx_mat::const_iterator it = A.begin();
    sp_cx_mat::const_iterator it_end = A.end();

    int nnz = 0;
    for (it; it != it_end; ++it)
    {
        S[it.row()][it.col()] = "•";
        nnz++;
    }

    // Finally, print the matrix to screen.
    cout << endl;
    for (int i = 0; i < A.n_rows; i++)
    {
        cout << "| ";
        for (int j = 0; j < A.n_cols; j++)
        {
            cout << S[i][j] << " ";
        }
        cout << "|\n";
    }

    cout << endl;
    cout << "matrix size: " << A.n_rows << "x" << A.n_cols << endl;
    cout << "non-zero elements: " << nnz << endl;
    cout << endl;
}

int main()
{
    Matrix matrix;

    // Test case 1: M=5 (3x3 internal grid)
    std::cout << "Testing with M=5 (3x3 internal grid):" << std::endl;

    // Create a simple potential matrix
    arma::mat V1(3, 3, arma::fill::zeros);
    double h1 = 1.0 / 4.0; // since M=5
    double dt = 0.001;     // arbitrary choice for testing

    auto matrices1 = matrix.create_AB_matrices(V1, h1, dt, 5);

    std::cout << "\nMatrix A structure:" << std::endl;
    print_sp_matrix_structure(matrices1[0]);

    std::cout << "\nMatrix B structure:" << std::endl;
    print_sp_matrix_structure(matrices1[1]);

    // Test case 2: M=6 (4x4 internal grid)
    std::cout << "\nTesting with M=6 (4x4 internal grid):" << std::endl;

    // Create a simple potential matrix
    arma::mat V2(4, 4, arma::fill::zeros);
    double h2 = 1.0 / 5.0; // since M=6

    auto matrices2 = matrix.create_AB_matrices(V2, h2, dt, 6);

    std::cout << "\nMatrix A structure:" << std::endl;
    print_sp_matrix_structure(matrices2[0]);

    std::cout << "\nMatrix B structure:" << std::endl;
    print_sp_matrix_structure(matrices2[1]);

    // Test case 3: Testing Gauss-Seidel solver
    std::cout << "\nTesting Gauss-Seidel solver with M=5 (3x3 internal grid):" << std::endl;

    // Use matrices1[0] (A) and matrices1[1] (B) from the first test case
    arma::sp_cx_mat A = matrices1[0];
    arma::sp_cx_mat B = matrices1[1];

    // Create a test vector b with ones
    int n = (5 - 2) * (5 - 2); // Size is (M-2)^2
    arma::cx_vec b(n, arma::fill::ones);

    // Solve using Gauss-Seidel
    std::cout << "Solving A * x = b..." << std::endl;
    arma::cx_vec x = matrix.solve_gauss_seidel(A, b);

    // Verify solution
    arma::cx_vec residual = b - A * x;
    double relative_residual = arma::norm(residual) / arma::norm(b);

    std::cout << std::scientific << std::setprecision(3);
    std::cout << "Relative residual: " << relative_residual << std::endl;

    // Print solution vector
    std::cout << "\nSolution vector x:" << std::endl;
    for (int i = 0; i < n; ++i)
    {
        std::cout << "x[" << i << "] = " << x(i) << std::endl;
    }

    // Test case 4: Testing initial state creation
    std::cout << "\nTesting initial state creation:" << std::endl;

    // Parameters from project description
    int M = 201; // 200 steps -> 201 points
    double h = 1.0 / 200.0;
    double x_c = 0.25;
    double y_c = 0.5;
    double sigma_x = 0.05;
    double sigma_y = 0.05;
    double p_x = 200;
    double p_y = 0;

    arma::cx_mat U = matrix.create_initial_state(M, h, x_c, y_c,
                                                 sigma_x, sigma_y, p_x, p_y);

    // Verify normalization
    double total_prob = matrix.calculate_probability_sum(U);
    std::cout << "Total probability after normalization: " << total_prob << std::endl;

    // Print a few values
    std::cout << "\nSome values from the initial state:" << std::endl;
    std::cout << "U(100,100) = " << U(100, 100) << std::endl;
    std::cout << "U(50,100) = " << U(50, 100) << std::endl;
    std::cout << "U(0,0) = " << U(0, 0) << std::endl; // Should be 0 (boundary)

    // Test case 5: Testing potential creation
    std::cout << "\nTesting potential creation:" << std::endl;

    // Create potential
    M = 201;         // We already have M defined from earlier
    h = 1.0 / 200.0; // We already have h defined from earlier
    double v0 = 1e10;

    arma::mat V = matrix.create_potential(M, h, v0);

    // Print dimensions
    std::cout << "Potential matrix dimensions: " << V.n_rows << "x" << V.n_cols << std::endl;

    // Check wall center location
    int wall_center = round(0.5 / h) - 1;
    int y_center = (M - 2) / 2;
    std::cout << "\nChecking positions:" << std::endl;
    std::cout << "Wall center index: " << wall_center << std::endl;
    std::cout << "Y center index: " << y_center << std::endl;

    // Print wall values at key points
    std::cout << "Wall center value: " << V(wall_center, y_center) << std::endl;

    // Check slit positions
    int slit_sep = round(0.05 / h);
    int y_slit1 = y_center - slit_sep / 2;
    int y_slit2 = y_center + slit_sep / 2;
    std::cout << "Slit 1 value: " << V(wall_center, y_slit1) << std::endl;
    std::cout << "Slit 2 value: " << V(wall_center, y_slit2) << std::endl;

    // Print a vertical slice through the wall to visualize slit structure
    std::cout << "\nPotential values along y-axis at wall center (x=0.5):" << std::endl;
    for (int j = 90; j < 110; j++)
    {
        std::cout << "V(" << wall_center << "," << j << ") = " << V(wall_center, j) << std::endl;
    }

    // After existing test output, add:
    std::cout << "\nGrid measurements:" << std::endl;
    std::cout << "Wall thickness: " << round(0.02 / h) << " points" << std::endl;
    std::cout << "Slit aperture: " << round(0.05 / h) << " points" << std::endl;
    std::cout << "Slit separation: " << round(0.05 / h) << " points" << std::endl;

    // Print a wider range of values to see both slits
    std::cout << "\nPotential values around slits:" << std::endl;
    for (int j = y_center - 20; j <= y_center + 20; j++)
    {
        if (V(wall_center, j) > 0)
        {
            std::cout << "#"; // Wall
        }
        else
        {
            std::cout << " "; // Opening
        }
        if (j == y_center)
            std::cout << "|"; // Mark center
    }
    std::cout << std::endl;

    return 0;
}