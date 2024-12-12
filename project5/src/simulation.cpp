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

void Simulation::run(std::vector<int> save_timesteps) {
    int n_steps = round(T/dt);
    probability_history.reserve(n_steps + 1);
    
    // Store initial probability
    probability_history.push_back(arma::sum(arma::sum(get_probability_matrix())));
    
    // Sort save_timesteps to ensure we don't miss any
    std::sort(save_timesteps.begin(), save_timesteps.end());
    auto next_save = save_timesteps.begin();
    
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
        
        // Check if we need to save this timestep
        if (next_save != save_timesteps.end() && n == *next_save) {
            std::string prefix;
            if (n == 0) prefix = "initial";
            else if (n == n_steps-1) prefix = "final";
            else prefix = "middle";
            
            save_state(prefix, n);
            ++next_save;
        }
        
        // Store probability less frequently
        if (n % 100 == 0) {
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
    
    std::string prob_filename = filename + "_prob_" + time_str + ".bin";
    std::string real_filename = filename + "_real_" + time_str + ".bin";
    std::string imag_filename = filename + "_imag_" + time_str + ".bin";
    
    std::cout << "Attempting to save files:" << std::endl;
    std::cout << " - " << prob_filename << std::endl;
    std::cout << " - " << real_filename << std::endl;
    std::cout << " - " << imag_filename << std::endl;
    
    bool prob_saved = prob.save(prob_filename);
    bool real_saved = real_part.save(real_filename);
    bool imag_saved = imag_part.save(imag_filename);
    
    if (!prob_saved || !real_saved || !imag_saved) {
        std::cerr << "Error saving files!" << std::endl;
        std::cerr << "Probability saved: " << prob_saved << std::endl;
        std::cerr << "Real part saved: " << real_saved << std::endl;
        std::cerr << "Imaginary part saved: " << imag_saved << std::endl;
        
        // Check if matrices are empty or have NaN values
        std::cout << "Matrix sizes:" << std::endl;
        std::cout << "Probability: " << prob.n_rows << "x" << prob.n_cols << std::endl;
        std::cout << "Real part: " << real_part.n_rows << "x" << real_part.n_cols << std::endl;
        std::cout << "Imag part: " << imag_part.n_rows << "x" << imag_part.n_cols << std::endl;
        
        if (prob.has_nan() || real_part.has_nan() || imag_part.has_nan()) {
            std::cerr << "Warning: NaN values detected in matrices!" << std::endl;
        }
        
        throw std::runtime_error("Failed to save state files");
    }
    
    std::cout << "Successfully saved state files" << std::endl;
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