#include "IsingModel.hpp"
#include <fstream>
#include <iomanip>
#include <string>

void study_energy_distribution(int L, double T, int n_cycles, int n_samples, std::string filename) {
    // Create model
    IsingModel model(L, T);
    
    // Create data directory if it doesn't exist
    std::system("mkdir -p data");
    std::string full_path = "data/" + filename;
    std::ofstream outfile(full_path);
    
    outfile << std::setprecision(10);
    outfile << "sample,energy_per_spin\n";
    
    // Run several independent samples
    for(int sample = 0; sample < n_samples; sample++) {
        // Initialize with random state
        model.initialize_random();
        
        // Burn-in period (from Problem 5)
        for(int i = 0; i < 5000; i++) {
            model.monte_carlo_cycle();
        }
        
        // Collect samples after burn-in
        for(int cycle = 0; cycle < n_cycles; cycle++) {
            model.monte_carlo_cycle();
            outfile << sample << "," << model.get_energy_per_spin() << "\n";
        }
    }
    
    outfile.close();
    std::cout << "Data written to: " << full_path << std::endl;
}

int main() {
    int L = 20;              // Lattice size
    int n_cycles = 2000;     // Cycles per sample
    int n_samples = 25;      // Number of independent samples
    
    // Study T = 1.0 J/kB
    study_energy_distribution(L, 1.0, n_cycles, n_samples, "energy_dist_T1.0.csv");
    
    // Study T = 2.4 J/kB
    study_energy_distribution(L, 2.4, n_cycles, n_samples, "energy_dist_T2.4.csv");
    
    return 0;
}