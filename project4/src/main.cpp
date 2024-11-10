#include "IsingModel.hpp"
#include <iostream>
#include <iomanip>

// Function to run Monte Carlo simulation and get averages
void run_simulation(IsingModel& model, int n_cycles, 
                   double& avg_e, double& avg_m, 
                   double& avg_e2, double& avg_m2) {
    
    avg_e = avg_m = avg_e2 = avg_m2 = 0.0;
    
    for(int i = 0; i < n_cycles; i++) {
        model.monte_carlo_cycle();
        
        double e = model.get_energy_per_spin();
        double m = model.get_magnetization_per_spin();
        
        avg_e += e;
        avg_m += m;
        avg_e2 += e*e;
        avg_m2 += m*m;
    }
    
    // Calculate averages
    avg_e /= n_cycles;
    avg_m /= n_cycles;
    avg_e2 /= n_cycles;
    avg_m2 /= n_cycles;
}

int main() {
    // Set parameters
    int L = 2;              // Lattice size
    double T = 1.0;         // Temperature in units of J/kB
    double J = 1.0;         // Coupling constant
    unsigned seed = 1234;   // Random seed
    int n_cycles = 100000;  // Number of Monte Carlo cycles
    
    // Initialize model
    IsingModel model(L, T, J, seed);
    
    // Test both random and ordered initial states
    std::cout << std::setprecision(6) << std::fixed;
    std::cout << "Testing L=" << L << " lattice at T=" << T << "J/kB\n" << std::endl;
    
    std::cout << "Starting from random configuration:" << std::endl;
    model.initialize_random();
    double avg_e, avg_m, avg_e2, avg_m2;
    run_simulation(model, n_cycles, avg_e, avg_m, avg_e2, avg_m2);
    
    // Calculate specific heat capacity and susceptibility
    double Cv = (avg_e2 - avg_e*avg_e)/(T*T);
    double chi = (avg_m2 - avg_m*avg_m)/T;
    
    std::cout << "Energy per spin: " << avg_e << " J" << std::endl;
    std::cout << "Magnetization per spin: " << avg_m << std::endl;
    std::cout << "Specific heat capacity per spin: " << Cv << " J/kB" << std::endl;
    std::cout << "Susceptibility per spin: " << chi << std::endl;
    
    std::cout << "\nStarting from ordered configuration:" << std::endl;
    model = IsingModel(L, T, J, seed);  // Reset model
    model.initialize_ordered();
    run_simulation(model, n_cycles, avg_e, avg_m, avg_e2, avg_m2);
    
    // Recalculate Cv and chi
    Cv = (avg_e2 - avg_e*avg_e)/(T*T);
    chi = (avg_m2 - avg_m*avg_m)/T;
    
    std::cout << "Energy per spin: " << avg_e << " J" << std::endl;
    std::cout << "Magnetization per spin: " << avg_m << std::endl;
    std::cout << "Specific heat capacity per spin: " << Cv << " J/kB" << std::endl;
    std::cout << "Susceptibility per spin: " << chi << std::endl;
    
    return 0;
}