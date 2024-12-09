// simulation.cpp
#include "simulation.hpp"

Simulation::Simulation(double h, double dt, double T,
                     double x_c, double y_c, 
                     double sigma_x, double sigma_y,
                     double p_x, double p_y,
                     double v0)
    : h(h), dt(dt), T(T)
{
    // Calculate grid size
    M = round(1.0/h) + 1;
    
    // Initialize potential
    V = matrix_solver.create_potential(M, h, v0);
    
    // Initialize wave function
    U = matrix_solver.create_initial_state(M, h, x_c, y_c, 
                                         sigma_x, sigma_y, p_x, p_y);
    
    // Create Crank-Nicolson matrices
    std::vector<arma::sp_cx_mat> matrices = 
        matrix_solver.create_AB_matrices(V, h, dt, M);
    A = matrices[0];
    B = matrices[1];
}

arma::cx_vec Simulation::state_to_vector() {
    // Convert internal points of U to vector form
    arma::cx_vec vec((M-2)*(M-2));
    for (int i = 0; i < M-2; i++) {
        for (int j = 0; j < M-2; j++) {
            vec(Matrix::to_single_index(i, j, M-2)) = U(i+1, j+1);
        }
    }
    return vec;
}

void Simulation::vector_to_state(const arma::cx_vec& vec) {
    // Convert vector back to matrix form (internal points only)
    for (int i = 0; i < M-2; i++) {
        for (int j = 0; j < M-2; j++) {
            U(i+1, j+1) = vec(Matrix::to_single_index(i, j, M-2));
        }
    }
}

void Simulation::run() {
    int n_steps = round(T/dt);
    
    for (int n = 0; n < n_steps; n++) {
        // Convert current state to vector form
        arma::cx_vec u_vec = state_to_vector();
        
        // Calculate B*u^n
        arma::cx_vec b = B * u_vec;
        
        // Solve A*u^(n+1) = b
        arma::cx_vec u_next = matrix_solver.solve_gauss_seidel(A, b);
        
        // Update state
        vector_to_state(u_next);
    }
}

arma::mat Simulation::get_probability() const {
    return arma::real(U % arma::conj(U));
}

double Simulation::get_total_probability() const {
    return arma::accu(get_probability());
}

void Simulation::save_state(const std::string& filename, int timestep) {
    // Save probability, real and imaginary parts
    arma::mat prob = get_probability();
    arma::mat real_part = get_real_part();
    arma::mat imag_part = get_imag_part();
    
    prob.save(filename + "_prob_" + std::to_string(timestep) + ".bin");
    real_part.save(filename + "_real_" + std::to_string(timestep) + ".bin");
    imag_part.save(filename + "_imag_" + std::to_string(timestep) + ".bin");
}