#include <iostream>
#include "PenningTrap.hpp"
#include "constants.hpp"

int main() {
    PenningTrap trap(1.0, 2.41e6, 500);
    trap.random_init_particles(100);

    // Run simulations and analyze results

    return 0;
}