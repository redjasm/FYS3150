#include "PenningTrap.hpp"
#include "Constants.hpp"

PenningTrap::PenningTrap(double B0_in, double V0_in, double d_in)
    : B0(B0_in), V0(V0_in), d(d_in) {}

void PenningTrap::add_particle(const Particle& p_in) {
    particles.push_back(p_in);
}

arma::vec PenningTrap::external_E_field(const arma::vec& r, double t) const {
    // Implement electric field calculation
}

arma::vec PenningTrap::external_B_field(const arma::vec& r) const {
    // Implement magnetic field calculation
}

// ... Implement other methods ...

void PenningTrap::evolve_RK4(double dt) {
    // Implement RK4 method
}

int PenningTrap::count_particles_inside() const {
    // Implement particle counting
}

void PenningTrap::random_init_particles(int n) {
    // Implement random particle initialization
// }