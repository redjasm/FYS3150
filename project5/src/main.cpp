// main.cpp
#include "simulation.hpp"
#include <iostream>
#include <iomanip>

void run_problem7() {
    // Case 1: No barrier
    Simulation sim1(0.005, 2.5e-5, 0.008,  // h, dt, T
                   0.25, 0.5,              // x_c, y_c
                   0.05, 0.05,             // sigma_x, sigma_y
                   200, 0,                 // p_x, p_y
                   0);                     // v0 (no barrier)
    
    sim1.run();
    arma::vec prob_history1 = sim1.get_probability_history();
    
    // Case 2: With barrier
    Simulation sim2(0.005, 2.5e-5, 0.008,  // h, dt, T
                   0.25, 0.5,              // x_c, y_c
                   0.05, 0.10,             // sigma_x, sigma_y
                   200, 0,                 // p_x, p_y
                   1e10);                  // v0 (with barrier)
    
    sim2.run();
    arma::vec prob_history2 = sim2.get_probability_history();
    
    // Save results for plotting
    prob_history1.save("prob_history_no_barrier.bin");
    prob_history2.save("prob_history_with_barrier.bin");
}

void run_problem8() {
    Simulation sim(0.005, 2.5e-5, 0.002,  // h, dt, T
                  0.25, 0.5,              // x_c, y_c
                  0.05, 0.20,             // sigma_x, sigma_y
                  200, 0,                 // p_x, p_y
                  1e10);                  // v0
    
    // Run simulation and save at specified timesteps
    std::vector<int> save_steps = {0, 40, 80};  // initial, middle, final
    sim.run(save_steps);
}

void run_problem9() {
    // Parameters
    double h = 0.005;
    double dt = 2.5e-5;
    double T = 0.002;
    double detector_x = 0.8;
    
    // Run simulations for different slit configurations
    for (int n_slits : {1, 2, 3}) {
        Simulation sim(h, dt, T,
                      0.25, 0.5,    // x_c, y_c
                      0.05, 0.20,   // sigma_x, sigma_y
                      200, 0,       // p_x, p_y
                      1e10,         // v0
                      n_slits);     // number of slits
        
        sim.run();
        
        // Get detector probabilities
        arma::vec detector_prob = sim.get_detector_probabilities(detector_x);
        detector_prob.save("detector_prob_" + std::to_string(n_slits) + "slits.bin");
    }
}

int main() {
    std::cout << "Running Problem 7 (Probability Conservation)...\n";
    run_problem7();
    
    std::cout << "Running Problem 8 (Time Evolution)...\n";
    run_problem8();
    
    std::cout << "Running Problem 9 (Detector Analysis)...\n";
    run_problem9();
    
    return 0;
}