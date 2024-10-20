#include <iostream>
#include <fstream>
#include <armadillo>
#include "PenningTrap.hpp"

int main() {
    // Problem 8: Single particle simulation
    PenningTrap trap(9.65e1, 9.65, 500);
    Particle p1(1, 40.078, {20, 0, 20}, {0, 25, 0});
    trap.add_particle(p1);

    double total_time = 50.0;
    int n_steps = 4000;
    double dt = total_time / n_steps;

    std::ofstream outfile("build/single_particle.csv");
    outfile << "t,x,y,z\n";

    for (int i = 0; i <= n_steps; ++i) {
        double t = i * dt;
        outfile << t << "," << trap.particles[0].r(0) << "," << trap.particles[0].r(1) << "," << trap.particles[0].r(2) << "\n";
        trap.evolve_RK4(dt);
    }

    outfile.close();

    // Problem 8: Two particle simulation
    PenningTrap trap2(9.65e1, 9.65, 500);
    Particle p2(1, 40.078, {20, 0, 20}, {0, 25, 0});
    Particle p3(1, 40.078, {25, 25, 0}, {0, 40, 5});
    trap2.add_particle(p2);
    trap2.add_particle(p3);

    std::ofstream outfile2("build/two_particles.csv");
    outfile2 << "t,x1,y1,z1,x2,y2,z2\n";

    for (int i = 0; i <= n_steps; ++i) {
        double t = i * dt;
        outfile2 << t << "," << trap2.particles[0].r(0) << "," << trap2.particles[0].r(1) << "," << trap2.particles[0].r(2) << ","
                 << trap2.particles[1].r(0) << "," << trap2.particles[1].r(1) << "," << trap2.particles[1].r(2) << "\n";
        trap2.evolve_RK4(dt);
    }

    outfile2.close();

    return 0;
}