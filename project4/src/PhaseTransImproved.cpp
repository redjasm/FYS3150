#include "IsingModel.hpp"
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

// Helper function to create linearly spaced vector
std::vector<double> linspace(double start, double end, int num)
{
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; i++)
    {
        result[i] = start + step * i;
    }
    return result;
}

void study_phase_transitions(std::vector<int> lattice_sizes,
                             std::vector<double> temperatures,
                             int n_cycles,
                             int n_samples, // Number of independent runs
                             std::string filename)
{
    // Create output directories
    std::system("mkdir -p data");

    // Open output file
    std::string filepath = "data/" + filename + ".csv";
    std::ofstream outfile(filepath);
    outfile << std::setprecision(10);

    // Write header
    outfile << "L,T,epsilon,m_abs,Cv,chi,sample" << std::endl;

    // Loop over lattice sizes
    for (int L : lattice_sizes)
    {
        std::cout << "Processing L = " << L << std::endl;

// Parallelize over temperatures
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (size_t i = 0; i < temperatures.size(); i++)
        {
            for (int sample = 0; sample < n_samples; sample++)
            {
                double T = temperatures[i];
                int thread_seed = omp_get_thread_num() + sample * 1000;

                // Initialize model
                IsingModel model(L, T);
                model.initialize_random();

                // Longer burn-in period
                for (int cycle = 0; cycle < 10000; cycle++)
                {
                    model.monte_carlo_cycle();
                }

                // Accumulate observables with more cycles
                double E_sum = 0.0, E2_sum = 0.0;
                double M_sum = 0.0, M2_sum = 0.0;

                for (int cycle = 0; cycle < n_cycles; cycle++)
                {
                    model.monte_carlo_cycle();

                    double E = model.get_energy_per_spin();
                    double M = std::abs(model.get_magnetization_per_spin());

                    E_sum += E;
                    E2_sum += E * E;
                    M_sum += M;
                    M2_sum += M * M;
                }

                // Calculate averages
                double N = static_cast<double>(n_cycles);
                double epsilon = E_sum / N;
                double m_abs = M_sum / N;

                // Calculate specific heat capacity and susceptibility
                double Cv = (E2_sum / N - epsilon * epsilon) / (T * T);
                double chi = (M2_sum / N - m_abs * m_abs) / T;

// Write results to file (thread-safe)
#pragma omp critical
                {
                    outfile << L << ","
                            << T << ","
                            << epsilon << ","
                            << m_abs << ","
                            << Cv << ","
                            << chi << ","
                            << sample << std::endl;
                }
            }
        }
    }

    outfile.close();
}

int main()
{
    // Lattice sizes to study
    std::vector<int> lattice_sizes = {40, 60, 80, 100};

    // More refined temperature ranges near expected Tc ≈ 2.269
    auto T1 = linspace(2.1, 2.24, 15);  // Coarse below
    auto T2 = linspace(2.24, 2.30, 30); // Fine near Tc
    auto T3 = linspace(2.30, 2.4, 15);  // Coarse above

    // Combine temperature ranges
    std::vector<double> temperatures;
    temperatures.insert(temperatures.end(), T1.begin(), T1.end());
    temperatures.insert(temperatures.end(), T2.begin(), T2.end());
    temperatures.insert(temperatures.end(), T3.begin(), T3.end());

    // Increase Monte Carlo cycles and add multiple samples
    int n_cycles = 100000; // 10x more cycles
    int n_samples = 5;     // 5 independent runs per temperature

    // Run simulation
    study_phase_transitions(lattice_sizes, temperatures, n_cycles, n_samples,
                            "phase_transition_improved");

    return 0;
}