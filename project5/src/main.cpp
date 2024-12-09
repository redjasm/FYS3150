#include "matrix.hpp"
#include <iostream>
#include <iomanip>

// Function to print the potential values around the slits
void print_potential_slice(const arma::mat& V, int wall_center, int y_center, int width = 20) {
    std::cout << "\nPotential values around slits (• indicates wall):" << std::endl;
    for (int j = y_center - width; j <= y_center + width; j++) {
        std::cout << (V(wall_center, j) > 0 ? "•" : " ");
        if (j == y_center) std::cout << "|"; // Mark center
    }
    std::cout << std::endl;
}

int main() {
    std::cout << std::scientific << std::setprecision(3);
    
    // Test potential creation with problem 5 parameters
    int M = 201;          // 200 steps -> 201 points
    double h = 1.0/200.0;
    double v0 = 1e10;
    
    Matrix matrix;
    arma::mat V = matrix.create_potential(M, h, v0);
    
    // Print dimensions and key parameters
    std::cout << "Potential matrix dimensions: " << V.n_rows << "x" << V.n_cols << std::endl;
    
    // Calculate and print important positions
    int wall_center = round(0.5/h) - 1;  // Wall at x = 0.5
    int y_center = (M-2)/2;              // Center in y-direction
    
    std::cout << "\nGrid measurements:" << std::endl;
    std::cout << "Wall center index: " << wall_center << std::endl;
    std::cout << "Y center index: " << y_center << std::endl;
    std::cout << "Wall thickness: " << round(0.02/h) << " points" << std::endl;
    std::cout << "Slit aperture: " << round(0.05/h) << " points" << std::endl;
    std::cout << "Slit separation: " << round(0.05/h) << " points" << std::endl;
    
    // Check key potential values
    std::cout << "\nPotential values at key points:" << std::endl;
    std::cout << "Wall center value: " << V(wall_center, y_center) << std::endl;
    
    // Calculate slit positions
    int slit_sep = round(0.05/h);
    int y_slit1 = y_center - slit_sep/2;
    int y_slit2 = y_center + slit_sep/2;
    
    std::cout << "Slit 1 value: " << V(wall_center, y_slit1) << std::endl;
    std::cout << "Slit 2 value: " << V(wall_center, y_slit2) << std::endl;
    
    // Print a slice of potential values
    std::cout << "\nPotential values along y-axis at wall center (x=0.5):" << std::endl;
    for (int j = y_center-10; j <= y_center+10; j++) {
        std::cout << "V(" << wall_center << "," << j << ") = " << V(wall_center, j) << std::endl;
    }
    
    // Print visual representation of the slits
    print_potential_slice(V, wall_center, y_center);
    
    return 0;
}