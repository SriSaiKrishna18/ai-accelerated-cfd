/**
 * @file main.cpp
 * @brief Main entry point for Navier-Stokes 2D solver
 * 
 * AI-HPC Hybrid Project - Mid Review
 * Demonstrates lid-driven cavity simulation with checkpoint output
 */

#include "solver.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void print_banner() {
    std::cout << "========================================\n";
    std::cout << " Navier-Stokes 2D Solver\n";
    std::cout << " AI-HPC Hybrid Project\n";
    std::cout << "========================================\n\n";
}

void ensure_directories() {
    // Create output directories using system calls (cross-platform)
#ifdef _WIN32
    system("mkdir data\\checkpoints 2>NUL");
    system("mkdir results 2>NUL");
#else
    system("mkdir -p data/checkpoints");
    system("mkdir -p results");
#endif
}

void print_parameters(const NavierStokesSolver::Parameters& params) {
    double Re = params.lx * 1.0 / params.nu;  // Reynolds number with lid velocity = 1
    double cfl_visc = params.nu * params.dt / (std::min(params.lx/(params.nx-1), 
                                                         params.ly/(params.ny-1)) * 
                                                std::min(params.lx/(params.nx-1), 
                                                         params.ly/(params.ny-1)));
    
    std::cout << "=== Simulation Parameters ===\n";
    std::cout << "Grid: " << params.nx << " x " << params.ny << "\n";
    std::cout << "Domain: " << params.lx << " x " << params.ly << "\n";
    std::cout << "Viscosity (nu): " << params.nu << "\n";
    std::cout << "Reynolds number: " << Re << "\n";
    std::cout << "Time step (dt): " << params.dt << "\n";
    std::cout << "Viscous CFL: " << cfl_visc << "\n";
    std::cout << "BC type: " << params.bc_type << "\n";
    std::cout << "IC type: " << params.ic_type << "\n";
    std::cout << "============================\n\n";
}

int main(int argc, char** argv) {
    print_banner();
    
    // Create output directories
    ensure_directories();
    
    // Default parameters - lid-driven cavity
    NavierStokesSolver::Parameters params;
    params.nx = 128;           // Grid resolution
    params.ny = 128;
    params.lx = 1.0;           // Domain size
    params.ly = 1.0;
    params.dt = 0.001;         // Time step
    params.nu = 0.01;          // Viscosity (Re = 100)
    params.rho = 1.0;          // Density
    params.max_iter_pressure = 500;   // Pressure solver iterations
    params.tolerance = 1e-5;          // Pressure solver tolerance
    params.bc_type = "lid_driven_cavity";
    params.ic_type = "lid_driven_cavity";
    
    // Simulation time to checkpoint
    double t_checkpoint = 1.0;
    
    // Parse command-line arguments (optional)
    if (argc >= 2) {
        t_checkpoint = std::atof(argv[1]);
    }
    if (argc >= 3) {
        params.nx = std::atoi(argv[2]);
        params.ny = params.nx;  // Square grid
    }
    if (argc >= 4) {
        params.nu = std::atof(argv[3]);
    }
    
    print_parameters(params);
    
    try {
        // Create solver
        std::cout << "Initializing solver...\n";
        NavierStokesSolver solver(params);
        solver.initialize();
        
        // Run simulation
        std::cout << "Running simulation until t = " << t_checkpoint << "...\n\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        solver.solve_until(t_checkpoint);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Print summary
        std::cout << "\n========================================\n";
        std::cout << "Checkpoint reached!\n";
        std::cout << "----------------------------------------\n";
        std::cout << "Final time: " << solver.state().time << "\n";
        std::cout << "Total steps: " << solver.state().step << "\n";
        std::cout << "Runtime: " << duration.count() << " ms\n";
        std::cout << "Time per step: " << (double)duration.count() / solver.state().step << " ms\n";
        std::cout << "Final divergence: " << solver.compute_divergence() << "\n";
        std::cout << "Final kinetic energy: " << solver.compute_kinetic_energy() << "\n";
        std::cout << "========================================\n\n";
        
        // Save outputs
        std::cout << "Saving outputs...\n";
        solver.save_checkpoint("data/checkpoints/checkpoint.bin");
        solver.save_vtk("results/checkpoint.vtk");
        
        std::cout << "\n=== Simulation Complete ===\n";
        std::cout << "Checkpoint: data/checkpoints/checkpoint.bin\n";
        std::cout << "VTK file: results/checkpoint.vtk\n";
        std::cout << "\nUse ParaView to visualize the VTK file.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
