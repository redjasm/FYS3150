#pragma once
#include <armadillo>

class Particle
{
public:
    double q;    // Charge
    double m;    // Mass
    arma::vec r; // Position
    arma::vec v; // Velocity

    Particle(double charge, double mass, arma::vec position, arma::vec velocity);
};