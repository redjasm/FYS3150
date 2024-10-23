#include <iostream>
#include <fstream>
#include <armadillo>
#include <chrono>
#include "PenningTrap.hpp"
#include "Constants.hpp"

// Analytical solution for single particle
arma::vec analytical_solution(double t, double x0, double v0, double z0,
                              double B0, double V0, double d, double q, double m)
{
    double omega_0 = q * B0 / m;
    double omega_z2 = 2 * q * V0 / (m * d * d);
    double omega_p = (omega_0 + std::sqrt(omega_0 * omega_0 - 2 * omega_z2)) / 2;
    double omega_m = (omega_0 - std::sqrt(omega_0 * omega_0 - 2 * omega_z2)) / 2;

    double Ap = (v0 + omega_m * x0) / (omega_m - omega_p);
    double Am = -(v0 + omega_p * x0) / (omega_m - omega_p);

    double x = Ap * std::cos(omega_p * t) + Am * std::cos(omega_m * t);
    double y = -Ap * std::sin(omega_p * t) - Am * std::sin(omega_m * t);
    double z = z0 * std::cos(std::sqrt(omega_z2) * t);

    return arma::vec({x, y, z});
}

// Function to calculate relative error
double calculate_relative_error(const arma::vec &numerical, const arma::vec &analytical)
{
    return arma::norm(numerical - analytical) / arma::norm(analytical);
}

int main()
{
    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // System parameters
    double B0 = 9.65e1;
    double V0 = 2.41e6;
    double d = 500;
    double q = Constants::q_Ca;
    double m = Constants::m_Ca;

    // Initial conditions
    arma::vec r0 = {20, 0, 20};
    arma::vec v0 = {0, 25, 0};

    // 1. Single Particle Error Analysis
    std::vector<int> n_steps = {4000, 8000, 16000, 32000};
    double total_time = 50.0;

    for (int n : n_steps)
    {
        double dt = total_time / n;
        PenningTrap trap(B0, V0, d);
        Particle p1(q, m, r0, v0);
        trap.add_particle(p1);

        // Output files for both methods
        std::ofstream fe_file("build/forward_euler_" + std::to_string(n) + ".csv");
        std::ofstream rk4_file("build/rk4_" + std::to_string(n) + ".csv");
        std::ofstream error_file("build/error_" + std::to_string(n) + ".csv");

        fe_file << "t,x,y,z,vx,vy,vz\n";
        rk4_file << "t,x,y,z,vx,vy,vz\n";
        error_file << "t,fe_error,rk4_error\n";

        // Run both methods in parallel
        PenningTrap trap_fe = trap; // Copy for Forward Euler

        for (int i = 0; i <= n; ++i)
        {
            double t = i * dt;
            arma::vec analytical = analytical_solution(t, r0(0), v0(1), r0(2), B0, V0, d, q, m);

            // Calculate errors
            double fe_error = calculate_relative_error(trap_fe.particles[0].r, analytical);
            double rk4_error = calculate_relative_error(trap.particles[0].r, analytical);

            // Save data
            fe_file << t << "," << trap_fe.particles[0].r(0) << ","
                    << trap_fe.particles[0].r(1) << "," << trap_fe.particles[0].r(2) << ","
                    << trap_fe.particles[0].v(0) << "," << trap_fe.particles[0].v(1) << ","
                    << trap_fe.particles[0].v(2) << "\n";

            rk4_file << t << "," << trap.particles[0].r(0) << ","
                     << trap.particles[0].r(1) << "," << trap.particles[0].r(2) << ","
                     << trap.particles[0].v(0) << "," << trap.particles[0].v(1) << ","
                     << trap.particles[0].v(2) << "\n";

            error_file << t << "," << fe_error << "," << rk4_error << "\n";

            // Evolve systems
            trap_fe.evolve_forward_Euler(dt, t);
            trap.evolve_RK4(dt, t);
        }

        fe_file.close();
        rk4_file.close();
        error_file.close();
    }

    // 2. Two Particle Analysis
    PenningTrap trap_two(B0, V0, d);
    Particle p1(q, m, r0, v0);
    Particle p2(q, m, {25, 25, 0}, {0, 40, 5});
    trap_two.add_particle(p1);
    trap_two.add_particle(p2);

    // With interactions
    std::ofstream two_particle_file("build/two_particles_with_interaction.csv");
    two_particle_file << "t,x1,y1,z1,vx1,vy1,vz1,x2,y2,z2,vx2,vy2,vz2\n";

    int n_steps_two = 4000;
    double dt_two = total_time / n_steps_two;

    for (int i = 0; i <= n_steps_two; ++i)
    {
        double t = i * dt_two;
        two_particle_file << t;
        for (const auto &p : trap_two.particles)
        {
            two_particle_file << "," << p.r(0) << "," << p.r(1) << "," << p.r(2)
                              << "," << p.v(0) << "," << p.v(1) << "," << p.v(2);
        }
        two_particle_file << "\n";
        trap_two.evolve_RK4(dt_two, t);
    }
    two_particle_file.close();

    // Without interactions
    trap_two.particles[0] = p1; // Reset particles
    trap_two.particles[1] = p2;
    trap_two.coulomb_interaction = false;

    std::ofstream two_particle_no_int_file("build/two_particles_without_interaction.csv");
    two_particle_no_int_file << "t,x1,y1,z1,vx1,vy1,vz1,x2,y2,z2,vx2,vy2,vz2\n";

    for (int i = 0; i <= n_steps_two; ++i)
    {
        double t = i * dt_two;
        two_particle_no_int_file << t;
        for (const auto &p : trap_two.particles)
        {
            two_particle_no_int_file << "," << p.r(0) << "," << p.r(1) << "," << p.r(2)
                                     << "," << p.v(0) << "," << p.v(1) << "," << p.v(2);
        }
        two_particle_no_int_file << "\n";
        trap_two.evolve_RK4(dt_two, t);
    }
    two_particle_no_int_file.close();

    // 3. Resonance Analysis
    std::vector<double> f_values = {0.1, 0.4, 0.7};
    double w_min = 0.2;
    double w_max = 2.5;
    double dw = 0.02;
    int n_particles = 100;
    total_time = 500.0;
    int n_steps_resonance = 1000;
    double dt = total_time / n_steps_resonance;

    for (double f : f_values)
    {
        std::cout << "Running resonance analysis for f = " << f << std::endl;
        std::ofstream res_file("build/resonance_f" + std::to_string(f) + ".csv");
        res_file << "w_V,particles_remaining\n";

        for (double w_V = w_min; w_V <= w_max; w_V += dw)
        {
            std::cout << "  Processing w_V = " << w_V << "/2.5" << std::endl;
            PenningTrap trap_res(B0, V0, d);
            trap_res.generate_random_particles(n_particles, q, m);
            trap_res.set_time_dependence(f, w_V);

            // Evolve system with fewer time steps
            for (int i = 0; i < n_steps_resonance; ++i)
            {
                if (i % 100 == 0)
                { // Print progress
                    std::cout << "    Time step: " << i << "/" << n_steps_resonance << "\r" << std::flush;
                }
                trap_res.evolve_RK4(dt, i * dt);
            }

            int remaining = trap_res.count_particles_inside();
            res_file << w_V << "," << remaining << "\n";
            std::cout << "    Particles remaining: " << remaining << "/" << n_particles << std::endl;
        }
        res_file.close();
    }

    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "\nTotal execution time: " << duration.count() << " seconds" << std::endl;

    return 0;

    return 0;
}