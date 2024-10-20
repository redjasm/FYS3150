#pragma once
#include <armadillo>
#include <vector>
#include "Particle.hpp"

class PenningTrap {
public:
    double B0, V0, d;
    std::vector<Particle> particles;
    bool coulomb_interaction;
    double f, w_V;  // Amplitude and angular frequency for time-dependent potential

    PenningTrap(double B0_in, double V0_in, double d_in);
    void add_particle(Particle p_in);
    arma::vec external_E_field(arma::vec r, double t);
    arma::vec external_B_field(arma::vec r);
    arma::vec force_particle(int i, int j);
    arma::vec total_force_external(int i, double t);
    arma::vec total_force_particles(int i);
    arma::vec total_force(int i, double t);
    void evolve_RK4(double dt, double t);
    void evolve_forward_Euler(double dt, double t);
    int count_particles_inside();
    void generate_random_particles(int n, double q, double m);
    void set_time_dependence(double f_in, double w_V_in);
};