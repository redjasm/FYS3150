#include "IsingModel.hpp"
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

void study_phase_transitions(std::vector<int> lattice_sizes, 
                           std::vector<double> temperatures, 
                           int n_cycles,
                           std::string filename) {
    // Create output directories
    std::system("mkdir -p data");
    
    // Open output file
    std::string filepath = "data/" + filename + ".csv";
    std::ofstream outfile(filepath);
    outfile << std::setprecision(10);
    
    // Write header
    outfile << "L,T,epsilon,m_abs,Cv,chi" << std::endl;
    
    // Loop over lattice sizes
    for (int L : lattice_sizes) {
        std::cout << "Processing L = " << L << std::endl;
        
        // Parallelize over temperatures
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < temperatures.size(); i++) {
            double T = temperatures[i];
            
            // Initialize model
            IsingModel model(L, T);
            model.initialize_random();
            
            // Burn-in period (from Problem 5)
            for (int cycle = 0; cycle < 5000; cycle++) {
                model.monte_carlo_cycle();
            }
            
            // Accumulate observables
            double E_sum = 0.0, E2_sum = 0.0;
            double M_sum = 0.0, M2_sum = 0.0;
            
            for (int cycle = 0; cycle < n_cycles; cycle++) {
                model.monte_carlo_cycle();
                
                double E = model.get_energy_per_spin();
                double M = std::abs(model.get_magnetization_per_spin());
                
                E_sum += E;
                E2_sum += E*E;
                M_sum += M;
                M2_sum += M*M;
            }
            
            // Calculate averages
            double N = static_cast<double>(n_cycles);
            double epsilon = E_sum/N;
            double m_abs = M_sum/N;
            
            // Calculate specific heat capacity and susceptibility
            double Cv = (E2_sum/N - epsilon*epsilon)/(T*T);
            double chi = (M2_sum/N - m_abs*m_abs)/T;
            
            // Write results to file (thread-safe)
            #pragma omp critical
            {
                outfile << L << ","
                       << T << ","
                       << epsilon << ","
                       << m_abs << ","
                       << Cv << ","
                       << chi << std::endl;
            }
        }
    }
    
    outfile.close();
}

std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; i++) {
        result[i] = start + step * i;
    }
    return result;
}

int main() {
    // Lattice sizes to study
    std::vector<int> lattice_sizes = {40, 60, 80, 100};
    
    // Temperature ranges
    auto T_coarse = linspace(2.1, 2.25, 10);
    auto T_fine = linspace(2.25, 2.35, 20);
    auto T_coarse2 = linspace(2.35, 2.4, 10);
    
    // Combine all temperature ranges
    std::vector<double> temperatures;
    temperatures.insert(temperatures.end(), T_coarse.begin(), T_coarse.end());
    temperatures.insert(temperatures.end(), T_fine.begin(), T_fine.end());
    temperatures.insert(temperatures.end(), T_coarse2.begin(), T_coarse2.end());
    
    // Monte Carlo cycles (after burn-in)
    int n_cycles = 10000;
    
    // Run simulation
    study_phase_transitions(lattice_sizes, temperatures, n_cycles, "phase_transition_data");
    
    return 0;
}