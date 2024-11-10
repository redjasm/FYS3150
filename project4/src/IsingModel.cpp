#include "IsingModel.hpp"

IsingModel::IsingModel(int size, double temp, double coupling, unsigned seed)
    : L(size), N(size*size), T(temp), J(coupling),
      lattice(size, size),
      rng(seed),
      uniform_real(0.0, 1.0),
      uniform_int(0, size-1)
{
    initialize_boltzmann_factors();
}

void IsingModel::initialize_boltzmann_factors() {
    // Pre-compute the possible Boltzmann factors
    // Energy changes can be: -8J, -4J, 0, 4J, 8J
    double beta = 1.0/T;  // Using units where kB = 1
    boltzmann_factors = {
        std::exp(8.0*beta*J),   // For ΔE = -8J
        std::exp(4.0*beta*J),   // For ΔE = -4J
        1.0,                    // For ΔE = 0
        std::exp(-4.0*beta*J),  // For ΔE = 4J
        std::exp(-8.0*beta*J)   // For ΔE = 8J
    };
}

void IsingModel::initialize_random() {
    // Fill lattice with random ±1 spins
    for(int i = 0; i < L; i++) {
        for(int j = 0; j < L; j++) {
            lattice(i,j) = uniform_real(rng) < 0.5 ? -1 : 1;
        }
    }
    total_energy = calculate_energy();
    total_magnetization = calculate_magnetization();
}

void IsingModel::initialize_ordered() {
    lattice.ones();  // All spins up
    total_energy = calculate_energy();
    total_magnetization = calculate_magnetization();
}

double IsingModel::calculate_energy() const {
    double E = 0.0;
    for(int i = 0; i < L; i++) {
        for(int j = 0; j < L; j++) {
            // Using periodic boundary conditions
            int right = (j + 1) % L;
            int down = (i + 1) % L;
            
            // Each spin interacts with right and down neighbors
            E += -J * lattice(i,j) * (lattice(i,right) + lattice(down,j));
        }
    }
    return E;
}

double IsingModel::calculate_magnetization() const {
    return arma::accu(lattice);
}

double IsingModel::calculate_energy_change(int i, int j) const {
    // Calculate ΔE for flipping spin (i,j)
    int left = (i - 1 + L) % L;
    int right = (i + 1) % L;
    int up = (j - 1 + L) % L;
    int down = (j + 1) % L;
    
    double sum_neighbors = lattice(left,j) + lattice(right,j) + 
                          lattice(i,up) + lattice(i,down);
    
    return 2.0 * J * lattice(i,j) * sum_neighbors;
}

void IsingModel::monte_carlo_cycle() {
    // Perform N=L×L spin flip attempts
    for(int n = 0; n < N; n++) {
        // Select random spin
        int i = uniform_int(rng);
        int j = uniform_int(rng);
        
        // Calculate energy change
        double dE = calculate_energy_change(i, j);
        
        // Get correct Boltzmann factor from lookup table
        int index = (dE + 8.0*J)/(4.0*J);  // Maps -8J→0, -4J→1, 0→2, 4J→3, 8J→4
        double acceptance = boltzmann_factors[index];
        
        // Accept/reject according to Metropolis algorithm
        if(uniform_real(rng) <= acceptance) {
            lattice(i,j) *= -1;  // Flip the spin
            total_energy += dE;
            total_magnetization += 2.0*lattice(i,j);
        }
    }
}