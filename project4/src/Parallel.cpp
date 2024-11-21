#include "IsingModel.hpp"
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/stat.h>

double run_simulation(int L, double T, int n_cycles, int n_threads) {
    omp_set_num_threads(n_threads);
    double start_time = omp_get_wtime();
    
    IsingModel model(L, T);
    model.initialize_random();
    
    for(int i = 0; i < 1000; i++) {
        model.monte_carlo_cycle();
    }
    
    #pragma omp parallel for
    for(int i = 0; i < n_cycles; i++) {
        model.monte_carlo_cycle();
    }
    
    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main() {
    int L = 100;
    double T = 2.4;
    int n_cycles = 10000;
    
    // Create output file
    std::ofstream outfile("data/parallel_timing.csv");
    outfile << "threads,time,speedup" << std::endl;
    
    // Run serial version
    double serial_time = run_simulation(L, T, n_cycles, 1);
    outfile << 1 << "," << serial_time << "," << 1.0 << std::endl;
    std::cout << "Serial time: " << serial_time << " seconds" << std::endl;
    
    // Test different thread counts
    std::vector<int> thread_counts = {2, 4, 8, 16};
    
    for(int n_threads : thread_counts) {
        double parallel_time = run_simulation(L, T, n_cycles, n_threads);
        double speedup = serial_time / parallel_time;
        
        outfile << n_threads << "," 
                << parallel_time << "," 
                << speedup << std::endl;
                
        std::cout << "Threads: " << n_threads 
                  << ", Time: " << parallel_time << " seconds"
                  << ", Speedup: " << speedup << "x" 
                  << std::endl;
    }
    
    outfile.close();
    return 0;
}