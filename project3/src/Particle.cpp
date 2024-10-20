#include "Particle.hpp"
#include <iostream>

Particle::Particle(double charge, double mass, arma::vec position, arma::vec velocity)
    : q(charge), m(mass), r(position), v(velocity), outside(false) {
    force = arma::vec(3, arma::fill::zeros);
}

arma::vec Particle::find_E_field(double V0, double d) const {
    arma::vec E = arma::vec(3).fill(0.);
    E(0) = this->r(0);
    E(1) = this->r(1);
    E(2) = -2*this->r(2);
    return E * V0 / (d*d);
}

arma::vec Particle::find_B_field(double B0) const{
    return {0, 0, B0};
}

arma::vec Particle::find_Lorentz_force(arma::vec E, arma::vec B) const {
    arma::vec F_E = this->q * E;
    arma::vec F_B = this->q * arma::cross(this->v, B);
    return F_E + F_B;
}

void Particle::print() const {
    std::cout << "Particle properties:" << std::endl;
    std::cout << "  Charge: " << q << std::endl;
    std::cout << "  Mass: " << m << std::endl;
    std::cout << "  Position: " << r.t() << std::endl;
    std::cout << "  Velocity: " << v.t() << std::endl;
    std::cout << "  Force: " << force.t() << std::endl;
    std::cout << "  Outside: " << (outside ? "Yes" : "No") << std::endl;
}

bool Particle::check_outside() const {
    return outside;
}

void Particle::is_outside() {
    outside = true;
}

void Particle::reset() {
    force.zeros();
    outside = false;
}