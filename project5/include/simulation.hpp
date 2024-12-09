// simulation.hpp
#ifndef SIMULATION_H
#define SIMULATION_H

#include <armadillo>
#include "matrix.hpp"

class Simulation
{
private:
    int M;                // Grid size
    double h;             // Spatial step size
    double dt;            // Time step
    double T;             // Total simulation time
    arma::cx_mat U;       // Current state
    arma::mat V;          // Potential
    Matrix matrix_solver; // Matrix operations handler
    arma::sp_cx_mat A, B; // Crank-Nicolson matrices

    // Helper functions
    arma::cx_vec state_to_vector();
    void vector_to_state(const arma::cx_vec &vec);

public:
    Simulation(double h, double dt, double T,
               double x_c, double y_c,
               double sigma_x, double sigma_y,
               double p_x, double p_y,
               double v0);

    void run();
    void save_state(const std::string &filename, int timestep);
    arma::mat get_probability() const;
    double get_total_probability() const;
    arma::mat get_real_part() const { return arma::real(U); }
    arma::mat get_imag_part() const { return arma::imag(U); }
};

#endif