/**
 * Parallel Navier-Stokes Solver using C++11 std::thread
 * 
 * This is a simplified parallel solver for benchmarking.
 * Uses std::thread instead of OpenMP for MinGW 6.3.0 compatibility.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <atomic>
#include <string>
#include <fstream>
#include "../include/parallel.h"

class ParallelSolver {
public:
    int nx, ny;
    double dx, dy, dt;
    double Re;  // Reynolds number
    
    std::vector<double> u, v, p;
    std::vector<double> u_star, v_star;
    std::vector<double> rhs;
    std::vector<double> p_old;
    
    int num_threads;
    
    ParallelSolver(int nx_, int ny_, double Re_ = 100.0, int threads = 0) 
        : nx(nx_), ny(ny_), Re(Re_) {
        
        num_threads = threads > 0 ? threads : std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        
        dx = 1.0 / (nx - 1);
        dy = 1.0 / (ny - 1);
        dt = 0.001;
        
        int size = nx * ny;
        u.resize(size, 0.0);
        v.resize(size, 0.0);
        p.resize(size, 0.0);
        u_star.resize(size, 0.0);
        v_star.resize(size, 0.0);
        rhs.resize(size, 0.0);
        p_old.resize(size, 0.0);
    }
    
    int idx(int i, int j) const { return j * nx + i; }
    
    // Initialize with lid-driven cavity
    void initialize() {
        // Set top boundary velocity (lid)
        for (int i = 0; i < nx; ++i) {
            u[idx(i, ny-1)] = 1.0;
        }
    }
    
    // Compute tentative velocity (parallel)
    void compute_tentative_velocity() {
        double nu = 1.0 / Re;
        double dx2 = dx * dx;
        double dy2 = dy * dy;
        
        // Parallel computation of u_star and v_star
        parallel_for_2d(ny - 2, nx - 2, [&](int jj, int ii) {
            int j = jj + 1;
            int i = ii + 1;
            
            // Convective terms
            double dudx = (u[idx(i+1,j)] - u[idx(i-1,j)]) / (2.0 * dx);
            double dudy = (u[idx(i,j+1)] - u[idx(i,j-1)]) / (2.0 * dy);
            double dvdx = (v[idx(i+1,j)] - v[idx(i-1,j)]) / (2.0 * dx);
            double dvdy = (v[idx(i,j+1)] - v[idx(i,j-1)]) / (2.0 * dy);
            
            // Diffusive terms
            double d2udx2 = (u[idx(i+1,j)] - 2.0*u[idx(i,j)] + u[idx(i-1,j)]) / dx2;
            double d2udy2 = (u[idx(i,j+1)] - 2.0*u[idx(i,j)] + u[idx(i,j-1)]) / dy2;
            double d2vdx2 = (v[idx(i+1,j)] - 2.0*v[idx(i,j)] + v[idx(i-1,j)]) / dx2;
            double d2vdy2 = (v[idx(i,j+1)] - 2.0*v[idx(i,j)] + v[idx(i,j-1)]) / dy2;
            
            // Update
            u_star[idx(i,j)] = u[idx(i,j)] + dt * (
                -u[idx(i,j)] * dudx - v[idx(i,j)] * dudy 
                + nu * (d2udx2 + d2udy2)
            );
            
            v_star[idx(i,j)] = v[idx(i,j)] + dt * (
                -u[idx(i,j)] * dvdx - v[idx(i,j)] * dvdy 
                + nu * (d2vdx2 + d2vdy2)
            );
        }, num_threads);
    }
    
    // Solve pressure Poisson (parallel Jacobi)
    void solve_pressure(int max_iter = 100) {
        double dx2 = dx * dx;
        double dy2 = dy * dy;
        double factor = 0.5 / (1.0/dx2 + 1.0/dy2);
        
        // Compute RHS
        parallel_for_2d(ny - 2, nx - 2, [&](int jj, int ii) {
            int j = jj + 1;
            int i = ii + 1;
            double dudx = (u_star[idx(i+1,j)] - u_star[idx(i-1,j)]) / (2.0 * dx);
            double dvdy = (v_star[idx(i,j+1)] - v_star[idx(i,j-1)]) / (2.0 * dy);
            rhs[idx(i,j)] = (dudx + dvdy) / dt;
        }, num_threads);
        
        // Jacobi iteration
        for (int iter = 0; iter < max_iter; ++iter) {
            // Copy p to p_old
            std::copy(p.begin(), p.end(), p_old.begin());
            
            // Update p (parallel)
            parallel_for_2d(ny - 2, nx - 2, [&](int jj, int ii) {
                int j = jj + 1;
                int i = ii + 1;
                p[idx(i,j)] = factor * (
                    (p_old[idx(i+1,j)] + p_old[idx(i-1,j)]) / dx2 +
                    (p_old[idx(i,j+1)] + p_old[idx(i,j-1)]) / dy2 -
                    rhs[idx(i,j)]
                );
            }, num_threads);
        }
    }
    
    // Project velocity (parallel)
    void project_velocity() {
        parallel_for_2d(ny - 2, nx - 2, [&](int jj, int ii) {
            int j = jj + 1;
            int i = ii + 1;
            double dpdx = (p[idx(i+1,j)] - p[idx(i-1,j)]) / (2.0 * dx);
            double dpdy = (p[idx(i,j+1)] - p[idx(i,j-1)]) / (2.0 * dy);
            u[idx(i,j)] = u_star[idx(i,j)] - dt * dpdx;
            v[idx(i,j)] = v_star[idx(i,j)] - dt * dpdy;
        }, num_threads);
    }
    
    // Apply boundary conditions
    void apply_boundary_conditions() {
        // Left and right boundaries (no-slip)
        for (int j = 0; j < ny; ++j) {
            u[idx(0, j)] = 0.0;
            v[idx(0, j)] = 0.0;
            u[idx(nx-1, j)] = 0.0;
            v[idx(nx-1, j)] = 0.0;
        }
        
        // Bottom boundary (no-slip)
        for (int i = 0; i < nx; ++i) {
            u[idx(i, 0)] = 0.0;
            v[idx(i, 0)] = 0.0;
        }
        
        // Top boundary (moving lid)
        for (int i = 0; i < nx; ++i) {
            u[idx(i, ny-1)] = 1.0;
            v[idx(i, ny-1)] = 0.0;
        }
    }
    
    // Single time step
    void step() {
        compute_tentative_velocity();
        solve_pressure(100);
        project_velocity();
        apply_boundary_conditions();
    }
    
    // Run simulation
    double run(int num_steps) {
        initialize();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int step = 0; step < num_steps; ++step) {
            this->step();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        return elapsed.count();
    }
};


// Benchmark function
void benchmark(int grid_size, int num_steps, int max_threads) {
    std::cout << "\n========================================\n";
    std::cout << "Grid: " << grid_size << "x" << grid_size << ", Steps: " << num_steps << "\n";
    std::cout << "========================================\n";
    
    double serial_time = 0.0;
    
    for (int threads = 1; threads <= max_threads; ++threads) {
        ParallelSolver solver(grid_size, grid_size, 100.0, threads);
        double time = solver.run(num_steps);
        
        if (threads == 1) {
            serial_time = time;
        }
        
        double speedup = serial_time / time;
        double efficiency = speedup / threads * 100.0;
        
        std::cout << "Threads: " << threads 
                  << " | Time: " << time << "s"
                  << " | Speedup: " << speedup << "x"
                  << " | Efficiency: " << efficiency << "%\n";
    }
}


int main(int argc, char* argv[]) {
    std::cout << "Navier-Stokes 2D - Parallel Solver Benchmark\n";
    std::cout << "Using std::thread (C++11)\n";
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << "\n";
    
    int max_threads = std::thread::hardware_concurrency();
    if (max_threads == 0) max_threads = 4;
    
    // Parse command line
    int grid_size = 64;
    int num_steps = 100;
    
    if (argc >= 2) grid_size = std::atoi(argv[1]);
    if (argc >= 3) num_steps = std::atoi(argv[2]);
    if (argc >= 4) max_threads = std::atoi(argv[3]);
    
    std::cout << "\nUsage: " << argv[0] << " [grid_size] [num_steps] [max_threads]\n";
    
    // Run benchmarks
    benchmark(grid_size, num_steps, max_threads);
    
    // Also test larger grids
    if (grid_size <= 64) {
        benchmark(128, 50, max_threads);
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Benchmark complete!\n";
    
    return 0;
}
