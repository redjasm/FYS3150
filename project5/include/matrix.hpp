#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <armadillo>
#include <complex>

class Matrix {
public:
    // Constructor with simulation parameters
    Matrix(int M, double h, double dt, const arma::cx_mat& V);
    
    // Build A and B matrices for Crank-Nicolson method
    void build(arma::sp_cx_mat& A, arma::sp_cx_mat& B);
    
    // Utility to print matrix structure
    static void print_structure(const arma::sp_cx_mat& matrix);

private:
    // Convert 2D indices to 1D index
    int idx(int i, int j) const;
    
    int M_;                    // Number of points in each dimension
    double dt_;               // Time step
    arma::cx_mat V_;          // Potential matrix
    std::complex<double> r_;  // Crank-Nicolson parameter i*dt/(2h^2)
};

#endif