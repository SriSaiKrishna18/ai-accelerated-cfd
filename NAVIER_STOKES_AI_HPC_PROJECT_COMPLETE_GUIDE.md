# Navier-Stokes 2D Fluid Flow: AI-HPC Hybrid Project
## Complete A-to-Z Implementation Guide

**Target**: Production-level project for FAANG applications  
**Budget**: $0 (using free/open-source tools only)  
**Timeline**: 4-6 weeks  
**Deliverables**: Optimized HPC solver + AI acceleration + Complete documentation

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Week 1: Foundation & Serial Implementation](#2-week-1-foundation--serial-implementation)
3. [Week 2: Parallel Optimization (OpenMP/MPI/CUDA)](#3-week-2-parallel-optimization)
4. [Week 3: AI Model Development](#4-week-3-ai-model-development)
5. [Week 4: Integration & Validation](#5-week-4-integration--validation)
6. [Week 5-6: Production Polish & Documentation](#6-week-5-6-production-polish--documentation)
7. [Daily Task Breakdown](#7-daily-task-breakdown)
8. [Code Repository Structure](#8-code-repository-structure)
9. [Testing Strategy](#9-testing-strategy)
10. [Presentation Materials](#10-presentation-materials)

---

## 1. Project Overview

### 1.1 What We're Building

A **hybrid computational fluid dynamics (CFD) solver** that combines:
- **HPC Component**: High-performance Navier-Stokes solver with OpenMP, MPI, and CUDA
- **AI Component**: Deep learning model (ConvLSTM/U-Net) for state prediction
- **Integration**: Seamless checkpoint → AI prediction → validation pipeline

### 1.2 Why This Stands Out for FAANG

✅ **Technical Depth**
- Multi-paradigm parallelization (shared memory, distributed, GPU)
- Production-quality C++17 code with modern CMake
- Deep learning applied to scientific computing

✅ **Engineering Excellence**
- Comprehensive testing and CI/CD
- Profiling and performance optimization
- Reproducible research practices

✅ **Problem-Solving**
- Bridging classical HPC with modern AI
- Real-world computational challenges
- Quantitative validation and error analysis

### 1.3 Key Technologies

**Languages**: C++17, Python 3.8+, CUDA C++  
**Parallel**: OpenMP 4.5+, MPI 3.1+, CUDA 11.0+  
**AI**: PyTorch 2.0+, ONNX  
**Tools**: CMake, Docker, GitHub Actions, Weights & Biases  
**Profiling**: gprof, perf, Nsight Compute

---

## 2. Week 1: Foundation & Serial Implementation

### Day 1-2: Environment Setup

#### Prerequisites Installation

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Core development tools
sudo apt-get install -y build-essential cmake git

# C++ libraries
sudo apt-get install -y libopenblas-dev libomp-dev

# Python environment
sudo apt-get install -y python3-pip python3-venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Python packages
pip install numpy scipy matplotlib h5py torch torchvision \
    wandb pytest black flake8 jupyter

# Optional: MPI (if not already installed)
sudo apt-get install -y mpich libmpich-dev

# Optional: CUDA (if you have NVIDIA GPU)
# Follow https://developer.nvidia.com/cuda-downloads
```

#### Project Structure Setup

```bash
mkdir -p navier-stokes-ai-hpc && cd navier-stokes-ai-hpc

# Create directory structure
mkdir -p src/{core,io,parallel} \
         include \
         tests \
         python/{models,data,training,visualization} \
         benchmarks \
         docs \
         scripts \
         data/{training,validation,checkpoints} \
         results

# Initialize git
git init
git branch -M main
```

#### `.gitignore`

```gitignore
# Build artifacts
build/
*.o
*.so
*.a
*.exe

# Python
__pycache__/
*.pyc
*.pyo
venv/
.env
*.egg-info/

# Data
data/training/*.h5
data/validation/*.h5
*.bin
*.dat

# IDE
.vscode/
.idea/
*.swp
*~

# Results
results/*.png
results/*.mp4
results/*.json

# Logs
*.log
wandb/
```

### Day 3-5: Serial Solver Implementation

#### File: `include/grid.h`

```cpp
#ifndef GRID_H
#define GRID_H

#include <vector>
#include <string>

class Grid {
public:
    Grid(int nx, int ny, double lx, double ly);
    
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    double lx() const { return lx_; }
    double ly() const { return ly_; }
    
    int index(int i, int j) const { return i + j * nx_; }
    int size() const { return nx_ * ny_; }
    
    void get_coords(int idx, int& i, int& j) const;
    double x(int i) const { return i * dx_; }
    double y(int j) const { return j * dy_; }
    
private:
    int nx_, ny_;
    double lx_, ly_;
    double dx_, dy_;
};

#endif // GRID_H
```

#### File: `src/core/grid.cpp`

```cpp
#include "grid.h"

Grid::Grid(int nx, int ny, double lx, double ly)
    : nx_(nx), ny_(ny), lx_(lx), ly_(ly) {
    dx_ = lx / (nx - 1);
    dy_ = ly / (ny - 1);
}

void Grid::get_coords(int idx, int& i, int& j) const {
    i = idx % nx_;
    j = idx / nx_;
}
```

#### File: `include/solver.h`

```cpp
#ifndef SOLVER_H
#define SOLVER_H

#include "grid.h"
#include <vector>
#include <string>
#include <memory>

class NavierStokesSolver {
public:
    struct Parameters {
        int nx, ny;
        double lx, ly;
        double dt;
        double nu;          // Kinematic viscosity
        double rho;         // Density
        int max_iter_pressure;
        double tolerance;
        std::string bc_type;
        std::string ic_type;
    };
    
    struct State {
        std::vector<double> u, v, p;
        double time;
        int step;
    };
    
    NavierStokesSolver(const Parameters& params);
    virtual ~NavierStokesSolver() = default;
    
    // Main simulation methods
    void initialize();
    virtual void step();
    void solve_until(double t_final);
    
    // I/O
    void save_checkpoint(const std::string& filename) const;
    void load_checkpoint(const std::string& filename);
    void save_vtk(const std::string& filename) const;
    
    // Accessors
    const State& state() const { return state_; }
    const Grid& grid() const { return *grid_; }
    const Parameters& params() const { return params_; }
    
    // Diagnostics
    double compute_divergence() const;
    double compute_kinetic_energy() const;
    double compute_cfl_number() const;
    
protected:
    Parameters params_;
    std::unique_ptr<Grid> grid_;
    State state_;
    
    // Work arrays
    std::vector<double> u_star_, v_star_;
    std::vector<double> rhs_;
    
    // Core methods
    virtual void compute_tentative_velocity();
    virtual void solve_pressure_poisson();
    virtual void project_velocity();
    void apply_boundary_conditions();
    
    // Initial conditions
    void set_lid_driven_cavity();
    void set_taylor_green();
    void set_shear_layer();
    void set_vortex_pair();
};

#endif // SOLVER_H
```

#### File: `src/core/solver.cpp`

```cpp
#include "solver.h"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>

NavierStokesSolver::NavierStokesSolver(const Parameters& params)
    : params_(params) {
    
    grid_ = std::make_unique<Grid>(params_.nx, params_.ny, 
                                    params_.lx, params_.ly);
    
    const int n = grid_->size();
    state_.u.resize(n, 0.0);
    state_.v.resize(n, 0.0);
    state_.p.resize(n, 0.0);
    state_.time = 0.0;
    state_.step = 0;
    
    u_star_.resize(n, 0.0);
    v_star_.resize(n, 0.0);
    rhs_.resize(n, 0.0);
}

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
        throw std::runtime_error("Unknown initial condition type");
    }
}

void NavierStokesSolver::set_lid_driven_cavity() {
    std::fill(state_.u.begin(), state_.u.end(), 0.0);
    std::fill(state_.v.begin(), state_.v.end(), 0.0);
    std::fill(state_.p.begin(), state_.p.end(), 0.0);
}

void NavierStokesSolver::set_taylor_green() {
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double x = grid_->x(i);
            double y = grid_->y(j);
            int idx = grid_->index(i, j);
            
            state_.u[idx] =  std::sin(x) * std::cos(y);
            state_.v[idx] = -std::cos(x) * std::sin(y);
            state_.p[idx] = -0.25 * (std::cos(2*x) + std::cos(2*y));
        }
    }
}

void NavierStokesSolver::set_shear_layer() {
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    const double delta = 0.05; // Shear layer thickness
    
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double y = grid_->y(j);
            int idx = grid_->index(i, j);
            
            if (y < 0.5) {
                state_.u[idx] = std::tanh((y - 0.25) / delta);
            } else {
                state_.u[idx] = std::tanh((0.75 - y) / delta);
            }
            
            // Add perturbation
            double x = grid_->x(i);
            state_.v[idx] = 0.05 * std::sin(2 * M_PI * x);
        }
    }
}

void NavierStokesSolver::set_vortex_pair() {
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    const double r0 = 0.15; // Vortex core radius
    
    // Two counter-rotating vortices
    double x1 = 0.25 * params_.lx;
    double y1 = 0.5 * params_.ly;
    double x2 = 0.75 * params_.lx;
    double y2 = 0.5 * params_.ly;
    double gamma = 1.0; // Circulation strength
    
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double x = grid_->x(i);
            double y = grid_->y(j);
            int idx = grid_->index(i, j);
            
            // First vortex (counter-clockwise)
            double dx1 = x - x1;
            double dy1 = y - y1;
            double r1 = std::sqrt(dx1*dx1 + dy1*dy1);
            double factor1 = gamma * (1 - std::exp(-r1*r1/(r0*r0))) / (2*M_PI*r1 + 1e-10);
            
            // Second vortex (clockwise)
            double dx2 = x - x2;
            double dy2 = y - y2;
            double r2 = std::sqrt(dx2*dx2 + dy2*dy2);
            double factor2 = -gamma * (1 - std::exp(-r2*r2/(r0*r0))) / (2*M_PI*r2 + 1e-10);
            
            state_.u[idx] = -factor1 * dy1 - factor2 * dy2;
            state_.v[idx] =  factor1 * dx1 + factor2 * dx2;
        }
    }
}

void NavierStokesSolver::compute_tentative_velocity() {
    const double dt = params_.dt;
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    const double nu = params_.nu;
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const int idx = grid_->index(i, j);
            
            const double u_ij = state_.u[idx];
            const double v_ij = state_.v[idx];
            
            // Advection: -u·∇u
            const double dudx = (state_.u[grid_->index(i+1, j)] - 
                                state_.u[grid_->index(i-1, j)]) / (2*dx);
            const double dudy = (state_.u[grid_->index(i, j+1)] - 
                                state_.u[grid_->index(i, j-1)]) / (2*dy);
            const double dvdx = (state_.v[grid_->index(i+1, j)] - 
                                state_.v[grid_->index(i-1, j)]) / (2*dx);
            const double dvdy = (state_.v[grid_->index(i, j+1)] - 
                                state_.v[grid_->index(i, j-1)]) / (2*dy);
            
            const double adv_u = u_ij * dudx + v_ij * dudy;
            const double adv_v = u_ij * dvdx + v_ij * dvdy;
            
            // Diffusion: ν∇²u
            const double d2udx2 = (state_.u[grid_->index(i+1, j)] - 2*u_ij + 
                                   state_.u[grid_->index(i-1, j)]) / (dx*dx);
            const double d2udy2 = (state_.u[grid_->index(i, j+1)] - 2*u_ij + 
                                   state_.u[grid_->index(i, j-1)]) / (dy*dy);
            const double d2vdx2 = (state_.v[grid_->index(i+1, j)] - 2*v_ij + 
                                   state_.v[grid_->index(i-1, j)]) / (dx*dx);
            const double d2vdy2 = (state_.v[grid_->index(i, j+1)] - 2*v_ij + 
                                   state_.v[grid_->index(i, j-1)]) / (dy*dy);
            
            const double diff_u = nu * (d2udx2 + d2udy2);
            const double diff_v = nu * (d2vdx2 + d2vdy2);
            
            // Tentative velocity
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
    
    // RHS: ρ/Δt ∇·u*
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const double dudx = (u_star_[grid_->index(i+1, j)] - 
                                u_star_[grid_->index(i-1, j)]) / (2*dx);
            const double dvdy = (v_star_[grid_->index(i, j+1)] - 
                                v_star_[grid_->index(i, j-1)]) / (2*dy);
            rhs_[grid_->index(i, j)] = (rho / dt) * (dudx + dvdy);
        }
    }
    
    // Jacobi iteration
    std::vector<double> p_old = state_.p;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double coeff = 1.0 / (2.0/dx2 + 2.0/dy2);
    
    for (int iter = 0; iter < params_.max_iter_pressure; ++iter) {
        double max_change = 0.0;
        
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
    
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const int idx = grid_->index(i, j);
            
            const double dpdx = (state_.p[grid_->index(i+1, j)] - 
                                state_.p[grid_->index(i-1, j)]) / (2*dx);
            const double dpdy = (state_.p[grid_->index(i, j+1)] - 
                                state_.p[grid_->index(i, j-1)]) / (2*dy);
            
            state_.u[idx] = u_star_[idx] - (dt / rho) * dpdx;
            state_.v[idx] = v_star_[idx] - (dt / rho) * dpdy;
        }
    }
}

void NavierStokesSolver::apply_boundary_conditions() {
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    if (params_.bc_type == "lid_driven_cavity") {
        // No-slip on all walls
        for (int i = 0; i < nx; ++i) {
            state_.u[grid_->index(i, 0)] = 0.0;
            state_.v[grid_->index(i, 0)] = 0.0;
            state_.u[grid_->index(i, ny-1)] = 1.0; // Lid velocity
            state_.v[grid_->index(i, ny-1)] = 0.0;
        }
        for (int j = 0; j < ny; ++j) {
            state_.u[grid_->index(0, j)] = 0.0;
            state_.v[grid_->index(0, j)] = 0.0;
            state_.u[grid_->index(nx-1, j)] = 0.0;
            state_.v[grid_->index(nx-1, j)] = 0.0;
        }
    } else if (params_.bc_type == "periodic") {
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
    
    // Neumann BC for pressure
    for (int i = 0; i < nx; ++i) {
        state_.p[grid_->index(i, 0)] = state_.p[grid_->index(i, 1)];
        state_.p[grid_->index(i, ny-1)] = state_.p[grid_->index(i, ny-2)];
    }
    for (int j = 0; j < ny; ++j) {
        state_.p[grid_->index(0, j)] = state_.p[grid_->index(1, j)];
        state_.p[grid_->index(nx-1, j)] = state_.p[grid_->index(nx-2, j)];
    }
}

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

double NavierStokesSolver::compute_divergence() const {
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    double max_div = 0.0;
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const double dudx = (state_.u[grid_->index(i+1, j)] - 
                                state_.u[grid_->index(i-1, j)]) / (2*dx);
            const double dvdy = (state_.v[grid_->index(i, j+1)] - 
                                state_.v[grid_->index(i, j-1)]) / (2*dy);
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

void NavierStokesSolver::save_checkpoint(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Write metadata
    int nx = grid_->nx();
    int ny = grid_->ny();
    file.write(reinterpret_cast<const char*>(&nx), sizeof(int));
    file.write(reinterpret_cast<const char*>(&ny), sizeof(int));
    file.write(reinterpret_cast<const char*>(&state_.time), sizeof(double));
    file.write(reinterpret_cast<const char*>(&state_.step), sizeof(int));
    
    // Write state
    file.write(reinterpret_cast<const char*>(state_.u.data()), 
               state_.u.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(state_.v.data()), 
               state_.v.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(state_.p.data()), 
               state_.p.size() * sizeof(double));
}

void NavierStokesSolver::load_checkpoint(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Read metadata
    int nx, ny;
    file.read(reinterpret_cast<char*>(&nx), sizeof(int));
    file.read(reinterpret_cast<char*>(&ny), sizeof(int));
    file.read(reinterpret_cast<char*>(&state_.time), sizeof(double));
    file.read(reinterpret_cast<char*>(&state_.step), sizeof(int));
    
    if (nx != grid_->nx() || ny != grid_->ny()) {
        throw std::runtime_error("Grid size mismatch");
    }
    
    // Read state
    file.read(reinterpret_cast<char*>(state_.u.data()), 
              state_.u.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(state_.v.data()), 
              state_.v.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(state_.p.data()), 
              state_.p.size() * sizeof(double));
}

void NavierStokesSolver::save_vtk(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "Navier-Stokes 2D\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << nx << " " << ny << " 1\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING " << grid_->dx() << " " << grid_->dy() << " 1\n";
    file << "POINT_DATA " << nx * ny << "\n";
    
    // Velocity
    file << "VECTORS velocity double\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int idx = grid_->index(i, j);
            file << state_.u[idx] << " " << state_.v[idx] << " 0\n";
        }
    }
    
    // Pressure
    file << "SCALARS pressure double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            file << state_.p[grid_->index(i, j)] << "\n";
        }
    }
}
```

#### File: `src/main.cpp`

```cpp
#include "solver.h"
#include <iostream>
#include <chrono>
#include <cmath>

int main(int argc, char** argv) {
    // Default parameters
    NavierStokesSolver::Parameters params;
    params.nx = 256;
    params.ny = 256;
    params.lx = 1.0;
    params.ly = 1.0;
    params.dt = 0.001;
    params.nu = 0.001;  // Reynolds = 1000
    params.rho = 1.0;
    params.max_iter_pressure = 1000;
    params.tolerance = 1e-6;
    params.bc_type = "lid_driven_cavity";
    params.ic_type = "lid_driven_cavity";
    
    double t_checkpoint = 1.0;
    
    std::cout << "========================================\n";
    std::cout << " Navier-Stokes 2D Solver\n";
    std::cout << "========================================\n";
    std::cout << "Grid: " << params.nx << " x " << params.ny << "\n";
    std::cout << "Reynolds number: " << params.lx / params.nu << "\n";
    std::cout << "CFL number: " << params.dt * 1.0 / std::min(params.lx/(params.nx-1), params.ly/(params.ny-1)) << "\n";
    std::cout << "========================================\n\n";
    
    try {
        NavierStokesSolver solver(params);
        solver.initialize();
        
        auto start = std::chrono::high_resolution_clock::now();
        solver.solve_until(t_checkpoint);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "\n========================================\n";
        std::cout << "Checkpoint reached!\n";
        std::cout << "Runtime: " << duration.count() << " ms\n";
        std::cout << "========================================\n";
        
        solver.save_checkpoint("data/checkpoints/checkpoint.bin");
        solver.save_vtk("results/checkpoint.vtk");
        
        std::cout << "Checkpoint saved.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

#### File: `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.15)
project(NavierStokesHPC VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_OPENMP "Enable OpenMP" OFF)
option(USE_MPI "Enable MPI" OFF)
option(USE_CUDA "Enable CUDA" OFF)
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_PROFILING "Enable profiling" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -pedantic")
endif()

if(ENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

# Source files
set(SOURCES
    src/core/grid.cpp
    src/core/solver.cpp
)

include_directories(include)

# Core library
add_library(nssolver STATIC ${SOURCES})

# OpenMP
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(nssolver PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(nssolver PUBLIC USE_OPENMP)
endif()

# MPI
if(USE_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(nssolver PUBLIC MPI::MPI_CXX)
    target_compile_definitions(nssolver PUBLIC USE_MPI)
endif()

# CUDA
if(USE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(nssolver PUBLIC USE_CUDA)
endif()

# Main executable
add_executable(ns_main src/main.cpp)
target_link_libraries(ns_main nssolver)

# Tests
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest)
    if(GTest_FOUND)
        add_subdirectory(tests)
    else()
        message(WARNING "Google Test not found, skipping tests")
    endif()
endif()

# Install
install(TARGETS ns_main nssolver
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
```

### Day 6-7: Testing & Validation

#### File: `tests/test_grid.cpp`

```cpp
#include <gtest/gtest.h>
#include "grid.h"

TEST(GridTest, BasicProperties) {
    Grid grid(100, 100, 1.0, 1.0);
    
    EXPECT_EQ(grid.nx(), 100);
    EXPECT_EQ(grid.ny(), 100);
    EXPECT_DOUBLE_EQ(grid.lx(), 1.0);
    EXPECT_DOUBLE_EQ(grid.ly(), 1.0);
    EXPECT_DOUBLE_EQ(grid.dx(), 1.0 / 99.0);
    EXPECT_DOUBLE_EQ(grid.dy(), 1.0 / 99.0);
}

TEST(GridTest, Indexing) {
    Grid grid(10, 10, 1.0, 1.0);
    
    EXPECT_EQ(grid.index(0, 0), 0);
    EXPECT_EQ(grid.index(9, 0), 9);
    EXPECT_EQ(grid.index(0, 9), 90);
    EXPECT_EQ(grid.index(9, 9), 99);
}

TEST(GridTest, Coordinates) {
    Grid grid(11, 11, 1.0, 1.0);
    
    EXPECT_DOUBLE_EQ(grid.x(0), 0.0);
    EXPECT_DOUBLE_EQ(grid.x(10), 1.0);
    EXPECT_DOUBLE_EQ(grid.y(0), 0.0);
    EXPECT_DOUBLE_EQ(grid.y(10), 1.0);
    EXPECT_DOUBLE_EQ(grid.x(5), 0.5);
    EXPECT_DOUBLE_EQ(grid.y(5), 0.5);
}
```

#### File: `tests/test_solver.cpp`

```cpp
#include <gtest/gtest.h>
#include "solver.h"
#include <cmath>

class SolverTest : public ::testing::Test {
protected:
    NavierStokesSolver::Parameters GetDefaultParams() {
        NavierStokesSolver::Parameters params;
        params.nx = 64;
        params.ny = 64;
        params.lx = 1.0;
        params.ly = 1.0;
        params.dt = 0.001;
        params.nu = 0.01;
        params.rho = 1.0;
        params.max_iter_pressure = 1000;
        params.tolerance = 1e-6;
        params.bc_type = "periodic";
        params.ic_type = "taylor_green";
        return params;
    }
};

TEST_F(SolverTest, Initialization) {
    auto params = GetDefaultParams();
    NavierStokesSolver solver(params);
    solver.initialize();
    
    const auto& state = solver.state();
    EXPECT_EQ(state.time, 0.0);
    EXPECT_EQ(state.step, 0);
}

TEST_F(SolverTest, DivergenceFree) {
    auto params = GetDefaultParams();
    NavierStokesSolver solver(params);
    solver.initialize();
    
    for (int i = 0; i < 100; ++i) {
        solver.step();
    }
    
    double div = solver.compute_divergence();
    EXPECT_LT(div, 1e-4) << "Divergence too large: " << div;
}

TEST_F(SolverTest, TaylorGreenDecay) {
    auto params = GetDefaultParams();
    params.nu = 0.01;
    params.bc_type = "periodic";
    params.ic_type = "taylor_green";
    
    NavierStokesSolver solver(params);
    solver.initialize();
    
    double ke0 = solver.compute_kinetic_energy();
    
    solver.solve_until(1.0);
    
    double ke1 = solver.compute_kinetic_energy();
    
    // Energy should decay
    EXPECT_LT(ke1, ke0);
    
    // Approximate decay rate: KE ~ exp(-2*nu*t)
    double expected_ke = ke0 * std::exp(-2 * params.nu * 1.0);
    EXPECT_NEAR(ke1, expected_ke, 0.2 * expected_ke);
}

TEST_F(SolverTest, CFLCondition) {
    auto params = GetDefaultParams();
    NavierStokesSolver solver(params);
    solver.initialize();
    
    solver.step();
    
    double cfl = solver.compute_cfl_number();
    EXPECT_LT(cfl, 1.0) << "CFL condition violated: " << cfl;
}

TEST_F(SolverTest, CheckpointSaveLoad) {
    auto params = GetDefaultParams();
    NavierStokesSolver solver1(params);
    solver1.initialize();
    
    solver1.solve_until(0.1);
    
    solver1.save_checkpoint("test_checkpoint.bin");
    
    NavierStokesSolver solver2(params);
    solver2.initialize();
    solver2.load_checkpoint("test_checkpoint.bin");
    
    const auto& state1 = solver1.state();
    const auto& state2 = solver2.state();
    
    EXPECT_DOUBLE_EQ(state1.time, state2.time);
    EXPECT_EQ(state1.step, state2.step);
    
    for (size_t i = 0; i < state1.u.size(); ++i) {
        EXPECT_DOUBLE_EQ(state1.u[i], state2.u[i]);
        EXPECT_DOUBLE_EQ(state1.v[i], state2.v[i]);
        EXPECT_DOUBLE_EQ(state1.p[i], state2.p[i]);
    }
}
```

#### File: `tests/CMakeLists.txt`

```cmake
# Test executable
add_executable(run_tests
    test_grid.cpp
    test_solver.cpp
)

target_link_libraries(run_tests
    nssolver
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME NavierStokesTests COMMAND run_tests)
```

#### Build and Run

```bash
# Create build directory
mkdir build && cd build

# Configure (Debug for testing)
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON ..

# Build
make -j$(nproc)

# Run tests
ctest --verbose

# OR run executable directly
./ns_main

# Check results
ls -lh ../data/checkpoints/
ls -lh ../results/
```

---

## 3. Week 2: Parallel Optimization

### Day 8-9: Profiling

```bash
# Enable profiling
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PROFILING=ON ..
make clean && make -j$(nproc)

# Run with profiling
./ns_main

# Analyze with gprof
gprof ./ns_main gmon.out > profile_analysis.txt

# Check hotspots
grep -A 20 "Flat profile" profile_analysis.txt

# Use perf for detailed analysis
perf record -g ./ns_main
perf report

# Cache analysis
perf stat -e cache-references,cache-misses,cycles,instructions ./ns_main
```

Create performance analysis script:

```python
# scripts/analyze_profile.py
import re
import matplotlib.pyplot as plt

def parse_gprof(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract function times
    pattern = r'(\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+\s+\d+\s+\d+\.\d+\s+\d+\.\d+\s+(\w+)'
    matches = re.findall(pattern, content)
    
    functions = {}
    for time, func in matches:
        functions[func] = float(time)
    
    return functions

def plot_profile(functions, output='profile.png'):
    # Sort by time
    sorted_funcs = sorted(functions.items(), key=lambda x: x[1], reverse=True)[:10]
    
    names = [f[0] for f in sorted_funcs]
    times = [f[1] for f in sorted_funcs]
    
    plt.figure(figsize=(12, 6))
    plt.barh(names, times)
    plt.xlabel('Time (%)')
    plt.title('Top 10 Hotspots')
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Profile plot saved to {output}")

if __name__ == "__main__":
    functions = parse_gprof('profile_analysis.txt')
    plot_profile(functions)
```

### Day 10-12: OpenMP Implementation

Create OpenMP versions of hot functions:

#### File: `src/parallel/solver_omp.cpp`

```cpp
#include "solver.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif

void NavierStokesSolver::compute_tentative_velocity() {
    const double dt = params_.dt;
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    const double nu = params_.nu;
    const int nx = grid_->nx();
    const int ny = grid_->ny();
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const int idx = grid_->index(i, j);
            
            // ... (same calculation as before)
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
    
    // RHS computation
    #pragma omp parallel for collapse(2)
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            // ... (same as before)
        }
    }
    
    // Red-Black Gauss-Seidel for better parallelization
    std::vector<double> p_old = state_.p;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double coeff = 1.0 / (2.0/dx2 + 2.0/dy2);
    
    for (int iter = 0; iter < params_.max_iter_pressure; ++iter) {
        double max_change = 0.0;
        
        // Red points (even i+j)
        #pragma omp parallel for collapse(2) reduction(max:max_change)
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                if ((i + j) % 2 == 0) {
                    const int idx = grid_->index(i, j);
                    double p_new = coeff * (
                        (state_.p[grid_->index(i+1, j)] + state_.p[grid_->index(i-1, j)]) / dx2 +
                        (state_.p[grid_->index(i, j+1)] + state_.p[grid_->index(i, j-1)]) / dy2 -
                        rhs_[idx]
                    );
                    max_change = std::max(max_change, std::abs(p_new - state_.p[idx]));
                    state_.p[idx] = p_new;
                }
            }
        }
        
        // Black points (odd i+j)
        #pragma omp parallel for collapse(2) reduction(max:max_change)
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                if ((i + j) % 2 == 1) {
                    const int idx = grid_->index(i, j);
                    double p_new = coeff * (
                        (state_.p[grid_->index(i+1, j)] + state_.p[grid_->index(i-1, j)]) / dx2 +
                        (state_.p[grid_->index(i, j+1)] + state_.p[grid_->index(i, j-1)]) / dy2 -
                        rhs_[idx]
                    );
                    max_change = std::max(max_change, std::abs(p_new - state_.p[idx]));
                    state_.p[idx] = p_new;
                }
            }
        }
        
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
    
    #pragma omp parallel for collapse(2)
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            // ... (same as before)
        }
    }
}
```

Benchmark OpenMP:

```bash
# Build with OpenMP
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON ..
make clean && make -j$(nproc)

# Test different thread counts
for threads in 1 2 4 8 16; do
    echo "Testing with $threads threads"
    export OMP_NUM_THREADS=$threads
    ./ns_main | grep "Runtime"
done
```

[*Due to length constraints, I'll provide the remaining content as a downloadable link or in continuation. The complete guide continues with MPI, CUDA, AI model training, integration, documentation, and presentation materials.*]

---

**CONTINUATION PREVIEW:**

## Remaining Sections (Week 3-6)

- **Week 3**: AI Model Development
  - Data generation pipeline
  - ConvLSTM/U-Net/PINN implementations
  - Training with PyTorch
  - Model optimization and quantization
  
- **Week 4**: Integration & Validation
  - Hybrid pipeline implementation
  - Error metrics and validation
  - Visualization tools
  - Performance comparison

- **Week 5-6**: Production Polish
  - Docker containerization
  - CI/CD with GitHub Actions
  - Comprehensive documentation
  - README with badges
  - Research paper draft
  - Presentation slides

Would you like me to:
1. Continue with the complete remaining content?
2. Focus on a specific section in detail?
3. Create separate files for each major component?

