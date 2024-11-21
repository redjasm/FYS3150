#include "IsingModel.hpp"
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>

// Helper function for linearly spaced temperatures
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

void study_phase_transitions(const std::vector<int> &lattice_sizes,
                             const std::vector<double> &temperatures,
                             int n_cycles,
                             const std::string &filename)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Open output file
    std::string filepath = "../data/" + filename + ".csv";
    std::ofstream outfile(filepath);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Could not open " << filepath << std::endl;
        return;
    }

    outfile << std::setprecision(10);
    outfile << "L,T,epsilon,m_abs,Cv,chi\n";

    // Track progress
    int total_computations = lattice_sizes.size() * temperatures.size();
    int completed_computations = 0;

    // Loop over lattice sizes
    for (int L : lattice_sizes)
    {
        std::cout << "\nProcessing L = " << L << std::endl;

// Parallelize over temperatures
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < temperatures.size(); i++)
        {
            double T = temperatures[i];

            // Initialize model with random state
            IsingModel model(L, T);
            model.initialize_random();

            // Burn-in period: ~5000 cycles seems sufficient from Problem 5
            for (int cycle = 0; cycle < 5000; cycle++)
            {
                model.monte_carlo_cycle();
            }

            // Accumulate observables
            double E_sum = 0.0, E2_sum = 0.0;
            double M_sum = 0.0, M2_sum = 0.0;

            // Main sampling loop
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

            // Calculate averages and derived quantities
            double N = static_cast<double>(n_cycles);
            double epsilon = E_sum / N;
            double m_abs = M_sum / N;
            double Cv = (E2_sum / N - epsilon * epsilon) / (T * T);
            double chi = (M2_sum / N - m_abs * m_abs) / T;

// Thread-safe write to file
#pragma omp critical
            {
                outfile << L << ","
                        << T << ","
                        << epsilon << ","
                        << m_abs << ","
                        << Cv << ","
                        << chi << std::endl;

                // Update and display progress
                completed_computations++;
                if (completed_computations % 10 == 0 ||
                    completed_computations == total_computations)
                {
                    double progress = 100.0 * completed_computations / total_computations;
                    std::cout << "\rProgress: " << std::fixed << std::setprecision(1)
                              << progress << "%" << std::flush;
                }
            }
        }
    }

    outfile.close();

    // Calculate and display total runtime
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "\n\nTotal runtime: " << duration.count() << " seconds" << std::endl;
    std::cout << "Data written to: " << filepath << std::endl;
}

int main()
{
    // Lattice sizes from project requirements
    std::vector<int> lattice_sizes = {40, 60, 80, 100};

    // Temperature grid focused around Tc ≈ 2.269 J/kB:
    // - Fewer points in stable regions (T << Tc and T >> Tc)
    // - More points near Tc for better resolution of transition
    auto T1 = linspace(2.1, 2.2, 5);  // Below Tc
    auto T2 = linspace(2.2, 2.3, 20); // Around Tc
    auto T3 = linspace(2.3, 2.4, 5);  // Above Tc

    // Combine temperature ranges
    std::vector<double> temperatures;
    temperatures.insert(temperatures.end(), T1.begin(), T1.end());
    temperatures.insert(temperatures.end(), T2.begin(), T2.end());
    temperatures.insert(temperatures.end(), T3.begin(), T3.end());

    // Number of Monte Carlo cycles:
    // - Need enough for good statistics but keep runtime reasonable
    // - 10000 cycles worked well in previous problems
    // - With 30 temperature points and 4 lattice sizes,
    //   this gives 120 total computations
    int n_cycles = 10000;

    // Run simulation
    study_phase_transitions(lattice_sizes, temperatures, n_cycles, "phase_transition_data");

    return 0;
}