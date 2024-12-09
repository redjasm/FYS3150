// test_matrix.cpp
#include "matrix.hpp"
#include <iostream>
#include <vector>
#include <string>

// A function that prints the structure of a sparse matrix to screen.
void print_sp_matrix_structure(const arma::sp_cx_mat& A)
{
    using namespace std;
    using namespace arma;

    // Declare a C-style 2D array of strings.
    string S[A.n_rows][A.n_cols];  

    // Initialise all the strings to " ".
    for (int i =0; i < A.n_rows; i++)
    {
        for (int j = 0; j < A.n_cols; j++)
        {
            S[i][j] = " ";
        }
    }

    // Next, we want to set the string to a dot at each non-zero element.
    // To do this we use the special loop iterator from the sp_cx_mat class
    // to help us loop over only the non-zero matrix elements.
    sp_cx_mat::const_iterator it     = A.begin();
    sp_cx_mat::const_iterator it_end = A.end();

    int nnz = 0;
    for(it; it != it_end; ++it)
    {
        S[it.row()][it.col()] = "•";
        nnz++;
    }

    // Finally, print the matrix to screen.
    cout << endl;
    for (int i =0; i < A.n_rows; i++)
    {
        cout << "| ";
        for (int j = 0; j < A.n_cols; j++)
        {
            cout << S[i][j] << " ";
        }
        cout <<  "|\n";
    }

    cout << endl;
    cout << "matrix size: " << A.n_rows << "x" << A.n_cols << endl;
    cout << "non-zero elements: " << nnz << endl ;
    cout << endl;
}

int main() {
    Matrix matrix;
    
    // Test case 1: M=5 (3x3 internal grid)
    std::cout << "Testing with M=5 (3x3 internal grid):" << std::endl;
    
    // Create a simple potential matrix
    arma::mat V1(3, 3, arma::fill::zeros);
    double h1 = 1.0/4.0;  // since M=5
    double dt = 0.001;    // arbitrary choice for testing
    
    auto matrices1 = matrix.create_AB_matrices(V1, h1, dt, 5);
    
    std::cout << "\nMatrix A structure:" << std::endl;
    print_sp_matrix_structure(matrices1[0]);
    
    std::cout << "\nMatrix B structure:" << std::endl;
    print_sp_matrix_structure(matrices1[1]);
    
    // Test case 2: M=6 (4x4 internal grid)
    std::cout << "\nTesting with M=6 (4x4 internal grid):" << std::endl;
    
    // Create a simple potential matrix
    arma::mat V2(4, 4, arma::fill::zeros);
    double h2 = 1.0/5.0;  // since M=6
    
    auto matrices2 = matrix.create_AB_matrices(V2, h2, dt, 6);
    
    std::cout << "\nMatrix A structure:" << std::endl;
    print_sp_matrix_structure(matrices2[0]);
    
    std::cout << "\nMatrix B structure:" << std::endl;
    print_sp_matrix_structure(matrices2[1]);
    
    return 0;
}