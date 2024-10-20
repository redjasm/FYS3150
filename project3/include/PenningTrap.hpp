#pragma once
#include <armadillo>
#include <vector>
#include "Particle.hpp"

class PenningTrap {
public:
    double B0, V0, d;
    std::vector<Particle> particles;

    PenningTrap(double B0_in, double V0_in, double d_in);
    void add_particle(Particle p_in);
    arma::vec external_E_field(arma::vec r);
    arma::vec external_B_field(arma::vec r);
    arma::vec force_particle(int i, int j);
    arma::vec total_force_external(int i);
    arma::vec total_force_particles(int i);
    arma::vec total_force(int i);
    void evolve_RK4(double dt);
    void evolve_forward_Euler(double dt);
};