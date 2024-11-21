#include "IsingModel.hpp"
#include <fstream>
#include <iomanip>

void study_burnin(int L, double T, int n_cycles, bool random_start, std::string filename) {
    IsingModel model(L, T);
    
    // Initialize grid
    if (random_start) {
        model.initialize_random();
    } else {
        model.initialize_ordered();
    }


    // Open output file with full path
    std::string full_path = "../data/" + filename;
    std::ofstream outfile(full_path);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << full_path << std::endl;
        return;
    }
    
    outfile << std::setprecision(10);
    outfile << "cycle,instant_energy,cumulative_energy\n";
    
    // Store initial values
    double energy_sum = model.get_energy_per_spin();
    outfile << 0 << "," << energy_sum << "," << energy_sum << "\n";
    
    // Run Monte Carlo cycles and record data
    for(int cycle = 1; cycle <= n_cycles; cycle++) {
        model.monte_carlo_cycle();
        
        double current_energy = model.get_energy_per_spin();
        energy_sum += current_energy;
        double avg_energy = energy_sum / (cycle + 1);
        
        outfile << cycle << "," << current_energy << "," << avg_energy << "\n";
    }
    
    outfile.close();
    std::cout << "Data written to: " << full_path << std::endl;
}

int main() {
    int L = 20;
    int n_cycles = 10000;
    
    // Study T = 1.0 J/kB
    study_burnin(L, 1.0, n_cycles, true, "burnin_T1.0_random.csv");
    study_burnin(L, 1.0, n_cycles, false, "burnin_T1.0_ordered.csv");
    
    // Study T = 2.4 J/kB
    study_burnin(L, 2.4, n_cycles, true, "burnin_T2.4_random.csv");
    study_burnin(L, 2.4, n_cycles, false, "burnin_T2.4_ordered.csv");
    
    return 0;
}