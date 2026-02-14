#include "solver.h"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Constructor
// =============================================================================

NavierStokesSolver::NavierStokesSolver(const Parameters& params)
    : params_(params) {
    
    // Create grid
    grid_ = std::make_unique<Grid>(params_.nx, params_.ny, 
                                    params_.lx, params_.ly);
    
    // Allocate state arrays
    const int n = grid_->size();
    state_.u.resize(n, 0.0);
    state_.v.resize(n, 0.0);
    state_.p.resize(n, 0.0);
    state_.time = 0.0;
    state_.step = 0;
    
    // Allocate work arrays
    u_star_.resize(n, 0.0);
    v_star_.resize(n, 0.0);
    rhs_.resize(n, 0.0);
}

// =============================================================================
// Initialization
// =============================================================================

void NavierStokesSolver::initialize() {
    if (params_.ic_type == "lid_driven_cavity") {
        set_lid_driven_cavity();
    } else if (params_.ic_type == "taylor_green") {
        set_taylor_green();
    } else if (params_.ic_type == "shear_layer") {
        set_shear_layer();
    } else if (params_.ic_type == "vortex_pair") {
        set_vortex_pair();
    } else {
        throw std::runtime_error("Unknown initial condition type: " + params_.ic_type);
    }
}

// =============================================================================
// Initial Conditions
// =============================================================================

void NavierStokesSolver::set_lid_driven_cavity() {
    // All velocities zero initially, lid velocity handled in BC
    std::fill(state_.u.begin(), state_.u.end(), 0.0);
    std::fill(state_.v.begin(), state_.v.end(), 0.0);
    std::fill(state_.p.begin(), state_.p.end(), 0.0);
}

void NavierStokesSolver::set_taylor_green() {
    // Taylor-Green vortex: analytical solution with viscous decay
    // u = sin(x) * cos(y) * exp(-2*nu*t)
    // v = -cos(x) * sin(y) * exp(-2*nu*t)
    // p = -0.25 * (cos(2x) + cos(2y)) * exp(-4*nu*t)
    
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double x = 2.0 * M_PI * grid_->x(i) / params_.lx;
            double y = 2.0 * M_PI * grid_->y(j) / params_.ly;
            int idx = grid_->index(i, j);
            
            state_.u[idx] =  std::sin(x) * std::cos(y);
            state_.v[idx] = -std::cos(x) * std::sin(y);
            state_.p[idx] = -0.25 * (std::cos(2*x) + std::cos(2*y));
        }
    }
}

void NavierStokesSolver::set_shear_layer() {
    // Shear layer with perturbation
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    const double delta = 0.05;  // Shear layer thickness
    
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double x = grid_->x(i);
            double y = grid_->y(j) / params_.ly;  // Normalized y
            int idx = grid_->index(i, j);
            
            // Hyperbolic tangent shear profile
            if (y < 0.5) {
                state_.u[idx] = std::tanh((y - 0.25) / delta);
            } else {
                state_.u[idx] = std::tanh((0.75 - y) / delta);
            }
            
            // Add perturbation to trigger instability
            state_.v[idx] = 0.05 * std::sin(2.0 * M_PI * x / params_.lx);
            state_.p[idx] = 0.0;
        }
    }
}

void NavierStokesSolver::set_vortex_pair() {
    // Co-rotating vortex pair
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    const double r0 = 0.15;  // Vortex core radius
    
    // Vortex centers
    double x1 = 0.25 * params_.lx;
    double y1 = 0.5 * params_.ly;
    double x2 = 0.75 * params_.lx;
    double y2 = 0.5 * params_.ly;
    double gamma = 1.0;  // Circulation strength
    
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double x = grid_->x(i);
            double y = grid_->y(j);
            int idx = grid_->index(i, j);
            
            // First vortex (counter-clockwise)
            double dx1 = x - x1;
            double dy1 = y - y1;
            double r1 = std::sqrt(dx1*dx1 + dy1*dy1);
            double factor1 = gamma * (1.0 - std::exp(-r1*r1/(r0*r0))) / (2.0*M_PI*r1 + 1e-10);
            
            // Second vortex (clockwise)
            double dx2 = x - x2;
            double dy2 = y - y2;
            double r2 = std::sqrt(dx2*dx2 + dy2*dy2);
            double factor2 = -gamma * (1.0 - std::exp(-r2*r2/(r0*r0))) / (2.0*M_PI*r2 + 1e-10);
            
            state_.u[idx] = -factor1 * dy1 - factor2 * dy2;
            state_.v[idx] =  factor1 * dx1 + factor2 * dx2;
            state_.p[idx] = 0.0;
        }
    }
}

// =============================================================================
// Core Numerical Methods
// =============================================================================

void NavierStokesSolver::compute_tentative_velocity() {
    const double dt = params_.dt;
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    const double nu = params_.nu;
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    // Copy boundary values
    for (int i = 0; i < nx; ++i) {
        u_star_[grid_->index(i, 0)] = state_.u[grid_->index(i, 0)];
        v_star_[grid_->index(i, 0)] = state_.v[grid_->index(i, 0)];
        u_star_[grid_->index(i, ny-1)] = state_.u[grid_->index(i, ny-1)];
        v_star_[grid_->index(i, ny-1)] = state_.v[grid_->index(i, ny-1)];
    }
    for (int j = 0; j < ny; ++j) {
        u_star_[grid_->index(0, j)] = state_.u[grid_->index(0, j)];
        v_star_[grid_->index(0, j)] = state_.v[grid_->index(0, j)];
        u_star_[grid_->index(nx-1, j)] = state_.u[grid_->index(nx-1, j)];
        v_star_[grid_->index(nx-1, j)] = state_.v[grid_->index(nx-1, j)];
    }
    
    // Interior points: advection-diffusion
    #ifdef USE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const int idx = grid_->index(i, j);
            
            const double u_ij = state_.u[idx];
            const double v_ij = state_.v[idx];
            
            // Central differences for advection
            const double dudx = (state_.u[grid_->index(i+1, j)] - 
                                state_.u[grid_->index(i-1, j)]) / (2.0*dx);
            const double dudy = (state_.u[grid_->index(i, j+1)] - 
                                state_.u[grid_->index(i, j-1)]) / (2.0*dy);
            const double dvdx = (state_.v[grid_->index(i+1, j)] - 
                                state_.v[grid_->index(i-1, j)]) / (2.0*dx);
            const double dvdy = (state_.v[grid_->index(i, j+1)] - 
                                state_.v[grid_->index(i, j-1)]) / (2.0*dy);
            
            const double adv_u = u_ij * dudx + v_ij * dudy;
            const double adv_v = u_ij * dvdx + v_ij * dvdy;
            
            // Laplacian for diffusion
            const double d2udx2 = (state_.u[grid_->index(i+1, j)] - 2.0*u_ij + 
                                   state_.u[grid_->index(i-1, j)]) / (dx*dx);
            const double d2udy2 = (state_.u[grid_->index(i, j+1)] - 2.0*u_ij + 
                                   state_.u[grid_->index(i, j-1)]) / (dy*dy);
            const double d2vdx2 = (state_.v[grid_->index(i+1, j)] - 2.0*v_ij + 
                                   state_.v[grid_->index(i-1, j)]) / (dx*dx);
            const double d2vdy2 = (state_.v[grid_->index(i, j+1)] - 2.0*v_ij + 
                                   state_.v[grid_->index(i, j-1)]) / (dy*dy);
            
            const double diff_u = nu * (d2udx2 + d2udy2);
            const double diff_v = nu * (d2vdx2 + d2vdy2);
            
            // Forward Euler update
            u_star_[idx] = u_ij + dt * (-adv_u + diff_u);
            v_star_[idx] = v_ij + dt * (-adv_v + diff_v);
        }
    }
}

void NavierStokesSolver::solve_pressure_poisson() {
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    const double dt = params_.dt;
    const double rho = params_.rho;
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    // Compute RHS: (rho/dt) * div(u*)
    #ifdef USE_OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const double dudx = (u_star_[grid_->index(i+1, j)] - 
                                u_star_[grid_->index(i-1, j)]) / (2.0*dx);
            const double dvdy = (v_star_[grid_->index(i, j+1)] - 
                                v_star_[grid_->index(i, j-1)]) / (2.0*dy);
            rhs_[grid_->index(i, j)] = (rho / dt) * (dudx + dvdy);
        }
    }
    
    // Jacobi iteration for pressure Poisson equation
    std::vector<double> p_old = state_.p;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double coeff = 1.0 / (2.0/dx2 + 2.0/dy2);
    
    for (int iter = 0; iter < params_.max_iter_pressure; ++iter) {
        double max_change = 0.0;
        
        #ifdef USE_OPENMP
        #pragma omp parallel for collapse(2) reduction(max:max_change)
        #endif
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                const int idx = grid_->index(i, j);
                const double p_new = coeff * (
                    (p_old[grid_->index(i+1, j)] + p_old[grid_->index(i-1, j)]) / dx2 +
                    (p_old[grid_->index(i, j+1)] + p_old[grid_->index(i, j-1)]) / dy2 -
                    rhs_[idx]
                );
                
                max_change = std::max(max_change, std::abs(p_new - state_.p[idx]));
                state_.p[idx] = p_new;
            }
        }
        
        p_old = state_.p;
        
        if (max_change < params_.tolerance) {
            break;
        }
    }
}

void NavierStokesSolver::project_velocity() {
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    const double dt = params_.dt;
    const double rho = params_.rho;
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    #ifdef USE_OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const int idx = grid_->index(i, j);
            
            const double dpdx = (state_.p[grid_->index(i+1, j)] - 
                                state_.p[grid_->index(i-1, j)]) / (2.0*dx);
            const double dpdy = (state_.p[grid_->index(i, j+1)] - 
                                state_.p[grid_->index(i, j-1)]) / (2.0*dy);
            
            state_.u[idx] = u_star_[idx] - (dt / rho) * dpdx;
            state_.v[idx] = v_star_[idx] - (dt / rho) * dpdy;
        }
    }
}

void NavierStokesSolver::apply_boundary_conditions() {
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    if (params_.bc_type == "lid_driven_cavity") {
        // No-slip on all walls, moving lid at top
        for (int i = 0; i < nx; ++i) {
            // Bottom wall: no-slip
            state_.u[grid_->index(i, 0)] = 0.0;
            state_.v[grid_->index(i, 0)] = 0.0;
            // Top wall: moving lid
            state_.u[grid_->index(i, ny-1)] = 1.0;  // Lid velocity
            state_.v[grid_->index(i, ny-1)] = 0.0;
        }
        for (int j = 0; j < ny; ++j) {
            // Left wall: no-slip
            state_.u[grid_->index(0, j)] = 0.0;
            state_.v[grid_->index(0, j)] = 0.0;
            // Right wall: no-slip
            state_.u[grid_->index(nx-1, j)] = 0.0;
            state_.v[grid_->index(nx-1, j)] = 0.0;
        }
    } else if (params_.bc_type == "periodic") {
        // Periodic in both directions
        for (int i = 0; i < nx; ++i) {
            state_.u[grid_->index(i, 0)] = state_.u[grid_->index(i, ny-2)];
            state_.v[grid_->index(i, 0)] = state_.v[grid_->index(i, ny-2)];
            state_.u[grid_->index(i, ny-1)] = state_.u[grid_->index(i, 1)];
            state_.v[grid_->index(i, ny-1)] = state_.v[grid_->index(i, 1)];
        }
        for (int j = 0; j < ny; ++j) {
            state_.u[grid_->index(0, j)] = state_.u[grid_->index(nx-2, j)];
            state_.v[grid_->index(0, j)] = state_.v[grid_->index(nx-2, j)];
            state_.u[grid_->index(nx-1, j)] = state_.u[grid_->index(1, j)];
            state_.v[grid_->index(nx-1, j)] = state_.v[grid_->index(1, j)];
        }
    }
    
    // Neumann BC for pressure (zero normal gradient)
    for (int i = 0; i < nx; ++i) {
        state_.p[grid_->index(i, 0)] = state_.p[grid_->index(i, 1)];
        state_.p[grid_->index(i, ny-1)] = state_.p[grid_->index(i, ny-2)];
    }
    for (int j = 0; j < ny; ++j) {
        state_.p[grid_->index(0, j)] = state_.p[grid_->index(1, j)];
        state_.p[grid_->index(nx-1, j)] = state_.p[grid_->index(nx-2, j)];
    }
}

// =============================================================================
// Time Stepping
// =============================================================================

void NavierStokesSolver::step() {
    compute_tentative_velocity();
    solve_pressure_poisson();
    project_velocity();
    apply_boundary_conditions();
    
    state_.time += params_.dt;
    state_.step += 1;
}

void NavierStokesSolver::solve_until(double t_final) {
    while (state_.time < t_final) {
        step();
        
        // Progress output every 100 steps
        if (state_.step % 100 == 0) {
            const double div = compute_divergence();
            const double ke = compute_kinetic_energy();
            const double cfl = compute_cfl_number();
            
            std::cout << "Step " << std::setw(6) << state_.step 
                      << " | t=" << std::setw(8) << std::fixed << std::setprecision(4) << state_.time
                      << " | div=" << std::scientific << std::setprecision(2) << div
                      << " | KE=" << std::fixed << std::setprecision(6) << ke
                      << " | CFL=" << std::fixed << std::setprecision(3) << cfl
                      << std::endl;
        }
    }
}

// =============================================================================
// Diagnostics
// =============================================================================

double NavierStokesSolver::compute_divergence() const {
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    double max_div = 0.0;
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const double dudx = (state_.u[grid_->index(i+1, j)] - 
                                state_.u[grid_->index(i-1, j)]) / (2.0*dx);
            const double dvdy = (state_.v[grid_->index(i, j+1)] - 
                                state_.v[grid_->index(i, j-1)]) / (2.0*dy);
            max_div = std::max(max_div, std::abs(dudx + dvdy));
        }
    }
    return max_div;
}

double NavierStokesSolver::compute_kinetic_energy() const {
    double ke = 0.0;
    for (size_t i = 0; i < state_.u.size(); ++i) {
        ke += state_.u[i] * state_.u[i] + state_.v[i] * state_.v[i];
    }
    return 0.5 * ke * grid_->dx() * grid_->dy();
}

double NavierStokesSolver::compute_cfl_number() const {
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    const double dt = params_.dt;
    
    double max_u = 0.0;
    double max_v = 0.0;
    for (size_t i = 0; i < state_.u.size(); ++i) {
        max_u = std::max(max_u, std::abs(state_.u[i]));
        max_v = std::max(max_v, std::abs(state_.v[i]));
    }
    
    return dt * (max_u / dx + max_v / dy);
}

// =============================================================================
// I/O
// =============================================================================

void NavierStokesSolver::save_checkpoint(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write metadata
    int nx = grid_->nx();
    int ny = grid_->ny();
    file.write(reinterpret_cast<const char*>(&nx), sizeof(int));
    file.write(reinterpret_cast<const char*>(&ny), sizeof(int));
    file.write(reinterpret_cast<const char*>(&state_.time), sizeof(double));
    file.write(reinterpret_cast<const char*>(&state_.step), sizeof(int));
    
    // Write state arrays
    file.write(reinterpret_cast<const char*>(state_.u.data()), 
               state_.u.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(state_.v.data()), 
               state_.v.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(state_.p.data()), 
               state_.p.size() * sizeof(double));
    
    std::cout << "Checkpoint saved: " << filename << std::endl;
}

void NavierStokesSolver::load_checkpoint(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // Read metadata
    int nx, ny;
    file.read(reinterpret_cast<char*>(&nx), sizeof(int));
    file.read(reinterpret_cast<char*>(&ny), sizeof(int));
    file.read(reinterpret_cast<char*>(&state_.time), sizeof(double));
    file.read(reinterpret_cast<char*>(&state_.step), sizeof(int));
    
    if (nx != grid_->nx() || ny != grid_->ny()) {
        throw std::runtime_error("Grid size mismatch in checkpoint");
    }
    
    // Read state arrays
    file.read(reinterpret_cast<char*>(state_.u.data()), 
              state_.u.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(state_.v.data()), 
              state_.v.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(state_.p.data()), 
              state_.p.size() * sizeof(double));
    
    std::cout << "Checkpoint loaded: " << filename << std::endl;
}

void NavierStokesSolver::save_vtk(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "Navier-Stokes 2D - Step " << state_.step << " Time " << state_.time << "\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << nx << " " << ny << " 1\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING " << grid_->dx() << " " << grid_->dy() << " 1\n";
    file << "POINT_DATA " << nx * ny << "\n";
    
    // Velocity vectors
    file << "VECTORS velocity double\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int idx = grid_->index(i, j);
            file << state_.u[idx] << " " << state_.v[idx] << " 0\n";
        }
    }
    
    // Pressure scalar
    file << "\nSCALARS pressure double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            file << state_.p[grid_->index(i, j)] << "\n";
        }
    }
    
    // Velocity magnitude
    file << "\nSCALARS velocity_magnitude double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int idx = grid_->index(i, j);
            double mag = std::sqrt(state_.u[idx]*state_.u[idx] + state_.v[idx]*state_.v[idx]);
            file << mag << "\n";
        }
    }
    
    std::cout << "VTK file saved: " << filename << std::endl;
}
