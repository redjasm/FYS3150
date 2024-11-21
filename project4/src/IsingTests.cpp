#include "IsingModel.hpp"
#include <iostream>
#include <iomanip>

void print_results(const IsingModel &model, int cycle)
{
    std::cout << std::setw(8) << cycle << " | "
              << std::setw(12) << std::fixed << std::setprecision(6)
              << model.get_energy_per_spin() << " | "
              << std::setw(12) << model.get_magnetization_per_spin() << " | "
              << std::setw(12) << model.get_energy_squared_per_spin() << " | "
              << std::setw(12) << model.get_magnetization_squared_per_spin()
              << std::endl;
}

int main()
{
    // Test parameters
    int L = 20;             // Lattice size
    double T = 1.0;         // Temperature
    double J = 1.0;         // Coupling constant
    int num_cycles = 10000; // Number of Monte Carlo cycles

    // Create model
    IsingModel model(L, T, J);

    std::cout << "Testing Ising Model with:" << std::endl
              << "Lattice size: " << L << "x" << L << std::endl
              << "Temperature: " << T << " J/kB" << std::endl
              << "Monte Carlo cycles: " << num_cycles << std::endl
              << std::endl;

    // Test with random initial state
    std::cout << "Starting from random configuration:" << std::endl;
    model.initialize_random();

    std::cout << std::setw(8) << "Cycle" << " | "
              << std::setw(12) << "Energy/N" << " | "
              << std::setw(12) << "Magnet/N" << " | "
              << std::setw(12) << "Energy²/N²" << " | "
              << std::setw(12) << "Magnet²/N²" << std::endl;

    std::cout << std::string(65, '-') << std::endl;

    // Print initial state
    print_results(model, 0);

    // Run simulation and print every 100 cycles
    for (int cycle = 1; cycle <= num_cycles; cycle++)
    {
        model.monte_carlo_cycle();
        if (cycle % 100 == 0)
        {
            print_results(model, cycle);
        }
    }

    std::cout << std::endl
              << "Testing with ordered initial state:" << std::endl;
    model.initialize_ordered();

    std::cout << std::string(65, '-') << std::endl;

    // Print initial state
    print_results(model, 0);

    // Run simulation and print every 100 cycles
    for (int cycle = 1; cycle <= num_cycles; cycle++)
    {
        model.monte_carlo_cycle();
        if (cycle % 100 == 0)
        {
            print_results(model, cycle);
        }
    }

    return 0;
}