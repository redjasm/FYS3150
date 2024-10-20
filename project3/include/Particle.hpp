#pragma once
#include <armadillo>

class Particle {
public:
    double q;  // Charge
    double m;  // Mass
    arma::vec r;  // Position
    arma::vec v;  // Velocity
    arma::vec force;  // Current force acting on the particle
    bool outside;  // Flag to indicate if particle is outside the trap

    // Constructor
    Particle(double charge, double mass, arma::vec position, arma::vec velocity);

    // Methods
    arma::vec find_E_field(double V0, double d) const;
    arma::vec find_B_field(double B0) const;
    arma::vec find_Lorentz_force(arma::vec E, arma::vec B) const;
    void print() const;
    bool check_outside() const;
    void is_outside();
    void reset();
};