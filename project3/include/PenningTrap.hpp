// #pragma once
#include <vector>
#include <armadillo>
#include "Particle.hpp"

class PenningTrap {
public:
    double B0;  // Magnetic field strength
    double V0;  // Applied potential
    double d;   // Characteristic dimension
    std::vector<Particle> particles;

    PenningTrap(double B0_in, double V0_in, double d_in);

    void add_particle(const Particle& p_in);
    arma::vec external_E_field(const arma::vec& r, double t = 0) const;
    arma::vec external_B_field(const arma::vec& r) const;
    arma::vec force_particle(int i, int j) const;
    arma::vec total_force_external(int i, double t = 0) const;
    arma::vec total_force_particles(int i) const;
    arma::vec total_force(int i, double t = 0) const;
    void evolve_RK4(double dt);
    void evolve_forward_Euler(double dt);
    int count_particles_inside() const;
    void random_init_particles(int n);
};