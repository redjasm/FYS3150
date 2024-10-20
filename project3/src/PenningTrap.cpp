#include "PenningTrap.hpp"
#include <cmath>
#include <random>

PenningTrap::PenningTrap(double B0_in, double V0_in, double d_in)
    : B0(B0_in), V0(V0_in), d(d_in), coulomb_interaction(true), f(0), w_V(0) {}

void PenningTrap::add_particle(Particle p_in) {
    particles.push_back(p_in);
}

arma::vec PenningTrap::external_E_field(arma::vec r, double t) {
    double V_t = V0 * (1 + f * std::cos(w_V * t));
    return (V_t / (d * d)) * arma::vec({r(0), r(1), -2 * r(2)});
}

arma::vec PenningTrap::external_B_field(arma::vec r) {
    return {0, 0, B0};
}

arma::vec PenningTrap::force_particle(int i, int j) {
    const double ke = 1.38935333e5;
    arma::vec r_ij = particles[i].r - particles[j].r;
    return ke * particles[i].q * particles[j].q * r_ij / std::pow(arma::norm(r_ij), 3);
}

arma::vec PenningTrap::total_force_external(int i, double t) {
    arma::vec E = external_E_field(particles[i].r, t);
    arma::vec B = external_B_field(particles[i].r);
    return particles[i].find_Lorentz_force(E, B);
}

arma::vec PenningTrap::total_force_particles(int i) {
    if (!coulomb_interaction) return arma::vec(3, arma::fill::zeros);
    
    arma::vec force(3, arma::fill::zeros);
    for (int j = 0; j < particles.size(); ++j) {
        if (i != j) {
            force += force_particle(i, j);
        }
    }
    return force;
}

arma::vec PenningTrap::total_force(int i, double t) {
    return total_force_external(i, t) + total_force_particles(i);
}

void PenningTrap::evolve_RK4(double dt, double t) {
    std::vector<Particle> original_particles = particles;

    for (int i = 0; i < particles.size(); ++i) {
        arma::vec k1_r = dt * particles[i].v;
        arma::vec k1_v = dt * total_force(i, t) / particles[i].m;

        particles[i].r += 0.5 * k1_r;
        particles[i].v += 0.5 * k1_v;

        arma::vec k2_r = dt * particles[i].v;
        arma::vec k2_v = dt * total_force(i, t + 0.5*dt) / particles[i].m;

        particles[i].r = original_particles[i].r + 0.5 * k2_r;
        particles[i].v = original_particles[i].v + 0.5 * k2_v;

        arma::vec k3_r = dt * particles[i].v;
        arma::vec k3_v = dt * total_force(i, t + 0.5*dt) / particles[i].m;

        particles[i].r = original_particles[i].r + k3_r;
        particles[i].v = original_particles[i].v + k3_v;

        arma::vec k4_r = dt * particles[i].v;
        arma::vec k4_v = dt * total_force(i, t + dt) / particles[i].m;

        particles[i].r = original_particles[i].r + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6;
        particles[i].v = original_particles[i].v + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6;
    }

    for (auto& p : particles) {
        if (arma::norm(p.r) > d) {
            p.is_outside();
        }
    }
}

void PenningTrap::evolve_forward_Euler(double dt, double t) {
    for (auto& p : particles) {
        p.r += dt * p.v;
        p.v += dt * total_force(&p - &particles[0], t) / p.m;
        
        if (arma::norm(p.r) > d) {
            p.is_outside();
        }
    }
}

int PenningTrap::count_particles_inside() {
    int count = 0;
    for (const auto& p : particles) {
        if (!p.check_outside()) {
            count++;
        }
    }
    return count;
}

void PenningTrap::generate_random_particles(int n, double q, double m) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1*d, 0.1*d);

    for (int i = 0; i < n; ++i) {
        arma::vec r = {dis(gen), dis(gen), dis(gen)};
        arma::vec v = {dis(gen), dis(gen), dis(gen)};
        particles.emplace_back(q, m, r, v);
    }
}

void PenningTrap::set_time_dependence(double f_in, double w_V_in) {
    f = f_in;
    w_V = w_V_in;
}