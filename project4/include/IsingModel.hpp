#ifndef ISING_MODEL_HPP
#define ISING_MODEL_HPP

#include <armadillo>
#include <random>

class IsingModel {
private:
    int L;                  // Lattice size
    int N;                  // Total number of spins (L×L)
    double T;              // Temperature in units of J/kB
    double J;              // Coupling constant
    arma::mat lattice;     // The spin lattice
    
    // System properties
    double total_energy;
    double total_magnetization;
    
    // Pre-computed Boltzmann factors for efficiency
    std::vector<double> boltzmann_factors;
    
    // Random number generators
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_real;
    std::uniform_int_distribution<int> uniform_int;
    
    // Private methods
    double calculate_energy() const;
    double calculate_magnetization() const;
    double calculate_energy_change(int i, int j) const;
    void initialize_boltzmann_factors();

public:
    // Constructor
    IsingModel(int size, double temp, double coupling=1.0, unsigned seed=1234);
    
    // Initialize lattice
    void initialize_random();    // Random configuration
    void initialize_ordered();   // All spins up
    
    // Monte Carlo methods
    void monte_carlo_cycle();
    
    // Getters for observables (per spin)
    double get_energy_per_spin() const { return total_energy/N; }
    double get_magnetization_per_spin() const { return std::abs(total_magnetization)/N; }
    double get_energy_squared_per_spin() const { return total_energy*total_energy/(N*N); }
    double get_magnetization_squared_per_spin() const { return total_magnetization*total_magnetization/(N*N); }
};

#endif