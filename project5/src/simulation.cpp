#include "simulation.hpp"

Simulation::Simulation(double h, double dt, double T,
                      double x_c, double y_c,
                      double sigma_x, double sigma_y,
                      double p_x, double p_y,
                      double v0, int n_slits)
    : h(h), dt(dt), T(T)
{
    // Calculate grid size
    M = round(1.0/h) + 1;
    
    // Initialize potential
    V = matrix_solver.create_potential(M, h, v0, n_slits);
    
    // Initialize wave function
    arma::cx_mat U_full = matrix_solver.create_initial_state(
        M, h, x_c, y_c, sigma_x, sigma_y, p_x, p_y);
    
    // Convert to vector form (internal points only)
    u = arma::cx_vec((M-2)*(M-2));
    for (int i = 0; i < M-2; i++) {
        for (int j = 0; j < M-2; j++) {
            u(Matrix::to_single_index(i, j, M-2)) = U_full(i+1, j+1);
        }
    }
    
    // Create Crank-Nicolson matrices
    std::vector<arma::sp_cx_mat> matrices = 
        matrix_solver.create_AB_matrices(V, h, dt, M);
    A = matrices[0];
    B = matrices[1];
}

void Simulation::run() {
    int n_steps = round(T/dt);
    probability_history.reserve(n_steps + 1);
    
    // Store initial probability
    probability_history.push_back(arma::sum(arma::sum(get_probability_matrix())));
    
    // Time evolution
    int progress_interval = std::max(1, n_steps / 10); 
    arma::cx_vec x;  // Reuse vector for solution
    
    for (int n = 0; n < n_steps; n++) {
        // Show progress
        if (n % progress_interval == 0) {
            std::cout << "Progress: " << (n * 100.0 / n_steps) << "%" << std::endl;
        }
        
        // Calculate B*u^n
        arma::cx_vec b = B * u;
        
        // Solve using LAPACK
        bool solved = arma::spsolve(x, A, b, "lapack");
        if (solved) {
            u = x;
        } else {
            std::cerr << "Warning: solver failed at step " << n << std::endl;
            // Fallback to Gauss-Seidel
            u = matrix_solver.solve_gauss_seidel(A, b);
        }
        
        // Store probability less frequently
        if (n % 100 == 0) {  // Reduced from every 10th to every 100th step
            probability_history.push_back(arma::sum(arma::sum(get_probability_matrix())));
        }
    }
    std::cout << "Simulation complete (100%)" << std::endl;
}

arma::mat Simulation::get_probability_matrix() const {
    arma::mat prob(M-2, M-2);
    for (int i = 0; i < M-2; i++) {
        for (int j = 0; j < M-2; j++) {
            std::complex<double> val = u(Matrix::to_single_index(i, j, M-2));
            prob(i,j) = std::norm(val);  // |z|^2 for complex z
        }
    }
    return prob;
}

arma::vec Simulation::get_detector_probabilities(double x_pos) const {
    // Convert x position to index
    int x_idx = round(x_pos/h) - 1;
    
    // Extract probabilities along y at fixed x
    arma::vec detector_prob(M-2);
    for (int j = 0; j < M-2; j++) {
        std::complex<double> val = u(Matrix::to_single_index(x_idx, j, M-2));
        detector_prob(j) = std::norm(val);
    }
    
    // Normalize
    detector_prob = detector_prob / arma::sum(detector_prob);
    
    return detector_prob;
}

arma::vec Simulation::get_probability_history() const {
    return arma::vec(probability_history);
}

void Simulation::save_state(const std::string& filename, int timestep) {
    // Get probability, real and imaginary parts
    arma::mat prob = get_probability_matrix();
    arma::mat real_part = get_real_part();
    arma::mat imag_part = get_imag_part();
    
    // Save each component with timestep in filename
    std::string time_str = std::to_string(timestep);
    prob.save(filename + "_prob_" + time_str + ".bin");
    real_part.save(filename + "_real_" + time_str + ".bin");
    imag_part.save(filename + "_imag_" + time_str + ".bin");
}

arma::mat Simulation::get_real_part() const {
    arma::mat real_part(M-2, M-2);
    for (int i = 0; i < M-2; i++) {
        for (int j = 0; j < M-2; j++) {
            real_part(i,j) = std::real(u(Matrix::to_single_index(i, j, M-2)));
        }
    }
    return real_part;
}

arma::mat Simulation::get_imag_part() const {
    arma::mat imag_part(M-2, M-2);
    for (int i = 0; i < M-2; i++) {
        for (int j = 0; j < M-2; j++) {
            imag_part(i,j) = std::imag(u(Matrix::to_single_index(i, j, M-2)));
        }
    }
    return imag_part;
}