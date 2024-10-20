#include <iostream>
#include <fstream>
#include <armadillo>
#include "PenningTrap.hpp"

int main() {
    PenningTrap trap(9.65e1, 9.65, 500);
    
    // Add two particles
    Particle p1(1, 40.078, {20, 0, 20}, {0, 25, 0});
    Particle p2(1, 40.078, {25, 25, 0}, {0, 40, 5});
    trap.add_particle(p1);
    trap.add_particle(p2);

    // Simulation parameters
    double total_time = 50.0;
    int n_steps = 4000;
    double dt = total_time / n_steps;

    // Run simulation with Coulomb interaction
    std::ofstream outfile("build/two_particles_with_interaction.csv");
    outfile << "t,x1,y1,z1,x2,y2,z2\n";

    for (int i = 0; i <= n_steps; ++i) {
        double t = i * dt;
        outfile << t << "," 
                << trap.particles[0].r(0) << "," << trap.particles[0].r(1) << "," << trap.particles[0].r(2) << ","
                << trap.particles[1].r(0) << "," << trap.particles[1].r(1) << "," << trap.particles[1].r(2) << "\n";
        trap.evolve_RK4(dt, t);
    }
    outfile.close();

    // Reset particles and run without Coulomb interaction
    trap.particles[0] = p1;
    trap.particles[1] = p2;
    trap.coulomb_interaction = false;

    outfile.open("build/two_particles_without_interaction.csv");
    outfile << "t,x1,y1,z1,x2,y2,z2\n";

    for (int i = 0; i <= n_steps; ++i) {
        double t = i * dt;
        outfile << t << "," 
                << trap.particles[0].r(0) << "," << trap.particles[0].r(1) << "," << trap.particles[0].r(2) << ","
                << trap.particles[1].r(0) << "," << trap.particles[1].r(1) << "," << trap.particles[1].r(2) << "\n";
        trap.evolve_RK4(dt, t);
    }
    outfile.close();

    // Simulation with random particles and time-dependent potential
    PenningTrap trap_random(9.65e1, 9.65, 500);
    trap_random.generate_random_particles(100, 1, 40.078);
    trap_random.set_time_dependence(0.1, 0.1);  // f = 0.1, w_V = 0.1

    outfile.open("build/random_particles_time_dependent.csv");
    outfile << "t,particles_inside\n";

    for (int i = 0; i <= n_steps; ++i) {
        double t = i * dt;
        outfile << t << "," << trap_random.count_particles_inside() << "\n";
        trap_random.evolve_RK4(dt, t);
    }
    outfile.close();

    return 0;
}