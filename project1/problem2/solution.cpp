#include <armadillo>
#include <iostream>
#include <fstream>
using namespace std; // so we don't have to write std:: everywhere

double u(double x) {
    return 1 - (1 - exp(-10)) * x - exp(-10 * x);
}

int main() {
    int n = 10000000;

    // n + 2 for the boundary conditions
    // Just n can cause out of bounds or undefined behavior
    arma::vec x = arma::linspace(0, 1, n+2);

    // using a matrix for efficient writing to file
    // Takes more memory, but is faster
    // Optionally use a vector and write to file in a loop
    arma::mat M(n, 2);

    for (int i = 0; i < n; i++) {
        M(i, 0) = x(i + 1);
        M(i, 1) = u(x(i + 1));
    }

    // out to solution_data.txt
    ofstream ofile;
    ofile.open("solution_data.txt");
    ofile << M;
    ofile.close();

    cout << "Solution data written to solution_data.txt" << endl;

    return 0;
}