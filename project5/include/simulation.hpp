#ifndef SIMULATION_H
#define SIMULATION_H

#include <armadillo>
#include "matrix.hpp"

class Simulation
{
private:
    // Core simulation parameters
    int M;     // Grid size
    double h;  // Spatial step size
    double dt; // Time step
    double T;  // Total simulation time

    // State and matrices
    arma::cx_vec u;       // Current state vector (internal points only)
    arma::mat V;          // Potential matrix
    arma::sp_cx_mat A, B; // Crank-Nicolson matrices
    Matrix matrix_solver; // Matrix operations handler

    // Storage for probability analysis
    std::vector<double> probability_history;

public:
    Simulation(double h, double dt, double T,
               double x_c, double y_c,
               double sigma_x, double sigma_y,
               double p_x, double p_y,
               double v0, int n_slits = 2);

    // Core simulation methods
    void run();
    void save_state(const std::string &filename, int timestep);

    // Analysis methods
    arma::mat get_probability_matrix() const;
    arma::vec get_probability_history() const;
    arma::vec get_detector_probabilities(double x_pos) const;

    // State conversion utilities
    arma::cx_mat get_state_matrix() const;
    arma::mat get_real_part() const;
    arma::mat get_imag_part() const;
};

#endif