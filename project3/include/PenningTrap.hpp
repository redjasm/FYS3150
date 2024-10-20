#define __Penningtrap__
#include <armadillo>
#include "Particle.hpp"

class PenningTrap {
public:
    double B0;  // Magnetic field strength
    double V0;  // Applied potential
    double d;   // Characteristic dimension
    std::vector<Particle> particles;

    PenningTrap(double B0_in, double V0_in, double d_in)
        : B0(B0_in), V0(V0_in), d(d_in) {}

    void add_particle(Particle p_in) {
        particles.push_back(p_in);
    }

    arma::vec external_E_field(arma::vec r) {
        // Implement electric field calculation
    }

    arma::vec external_B_field(arma::vec r) {
        // Implement magnetic field calculation
    }

    arma::vec force_particle(int i, int j) {
        // Implement particle interaction force
    }

    // ... (other methods as suggested in the problem description)
};