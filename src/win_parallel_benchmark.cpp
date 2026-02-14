/**
 * Windows-Native Parallel Solver Benchmark
 * 
 * Uses Windows threading API directly for MinGW 6.3.0 compatibility.
 * Falls back to serial if threading not available.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <string>

#ifdef _WIN32
#include <windows.h>
#define HAS_WINDOWS_THREADS 1
#else
#define HAS_WINDOWS_THREADS 0
#endif


// Grid class
class Grid {
public:
    int nx, ny;
    double dx, dy;
    std::vector<double> u, v, p;
    std::vector<double> u_star, v_star, rhs, p_old;
    
    Grid(int nx_, int ny_) : nx(nx_), ny(ny_) {
        dx = 1.0 / (nx - 1);
        dy = 1.0 / (ny - 1);
        
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
    
    double& U(int i, int j) { return u[idx(i,j)]; }
    double& V(int i, int j) { return v[idx(i,j)]; }
    double& P(int i, int j) { return p[idx(i,j)]; }
};


// Solver class
class Solver {
public:
    Grid grid;
    double Re, dt;
    int num_threads;
    bool use_parallel;
    
    Solver(int nx, int ny, double re = 100.0, int threads = 1) 
        : grid(nx, ny), Re(re), num_threads(threads) {
        dt = 0.001;
        use_parallel = (threads > 1) && HAS_WINDOWS_THREADS;
    }
    
    void initialize() {
        // Lid-driven cavity: top boundary velocity = 1
        for (int i = 0; i < grid.nx; ++i) {
            grid.U(i, grid.ny - 1) = 1.0;
        }
    }
    
    // Serial inner loop computation
    void compute_interior_point(int i, int j) {
        double nu = 1.0 / Re;
        double dx = grid.dx, dy = grid.dy;
        double dx2 = dx * dx, dy2 = dy * dy;
        
        // Convection
        double dudx = (grid.u[grid.idx(i+1,j)] - grid.u[grid.idx(i-1,j)]) / (2*dx);
        double dudy = (grid.u[grid.idx(i,j+1)] - grid.u[grid.idx(i,j-1)]) / (2*dy);
        double dvdx = (grid.v[grid.idx(i+1,j)] - grid.v[grid.idx(i-1,j)]) / (2*dx);
        double dvdy = (grid.v[grid.idx(i,j+1)] - grid.v[grid.idx(i,j-1)]) / (2*dy);
        
        // Diffusion
        double lapU = (grid.u[grid.idx(i+1,j)] - 2*grid.u[grid.idx(i,j)] + grid.u[grid.idx(i-1,j)]) / dx2
                    + (grid.u[grid.idx(i,j+1)] - 2*grid.u[grid.idx(i,j)] + grid.u[grid.idx(i,j-1)]) / dy2;
        double lapV = (grid.v[grid.idx(i+1,j)] - 2*grid.v[grid.idx(i,j)] + grid.v[grid.idx(i-1,j)]) / dx2
                    + (grid.v[grid.idx(i,j+1)] - 2*grid.v[grid.idx(i,j)] + grid.v[grid.idx(i,j-1)]) / dy2;
        
        grid.u_star[grid.idx(i,j)] = grid.u[grid.idx(i,j)] + dt * (
            -grid.u[grid.idx(i,j)] * dudx - grid.v[grid.idx(i,j)] * dudy + nu * lapU);
        grid.v_star[grid.idx(i,j)] = grid.v[grid.idx(i,j)] + dt * (
            -grid.u[grid.idx(i,j)] * dvdx - grid.v[grid.idx(i,j)] * dvdy + nu * lapV);
    }
    
    void compute_tentative_velocity_serial() {
        for (int j = 1; j < grid.ny - 1; ++j) {
            for (int i = 1; i < grid.nx - 1; ++i) {
                compute_interior_point(i, j);
            }
        }
    }

#if HAS_WINDOWS_THREADS
    struct ThreadData {
        Solver* solver;
        int j_start, j_end;
    };
    
    static DWORD WINAPI velocity_thread_func(LPVOID lpParam) {
        ThreadData* data = (ThreadData*)lpParam;
        for (int j = data->j_start; j < data->j_end; ++j) {
            for (int i = 1; i < data->solver->grid.nx - 1; ++i) {
                data->solver->compute_interior_point(i, j);
            }
        }
        return 0;
    }
    
    void compute_tentative_velocity_parallel() {
        int rows = grid.ny - 2;
        int rows_per_thread = (rows + num_threads - 1) / num_threads;
        
        std::vector<HANDLE> threads(num_threads);
        std::vector<ThreadData> data(num_threads);
        
        for (int t = 0; t < num_threads; ++t) {
            data[t].solver = this;
            data[t].j_start = 1 + t * rows_per_thread;
            data[t].j_end = std::min(data[t].j_start + rows_per_thread, grid.ny - 1);
            
            if (data[t].j_start >= grid.ny - 1) {
                threads[t] = NULL;
                continue;
            }
            
            threads[t] = CreateThread(NULL, 0, velocity_thread_func, &data[t], 0, NULL);
        }
        
        for (int t = 0; t < num_threads; ++t) {
            if (threads[t] != NULL) {
                WaitForSingleObject(threads[t], INFINITE);
                CloseHandle(threads[t]);
            }
        }
    }
#endif
    
    void compute_tentative_velocity() {
#if HAS_WINDOWS_THREADS
        if (use_parallel && num_threads > 1) {
            compute_tentative_velocity_parallel();
        } else {
            compute_tentative_velocity_serial();
        }
#else
        compute_tentative_velocity_serial();
#endif
    }
    
    void solve_pressure(int max_iter = 100) {
        double dx2 = grid.dx * grid.dx;
        double dy2 = grid.dy * grid.dy;
        double factor = 0.5 / (1.0/dx2 + 1.0/dy2);
        
        // Compute RHS
        for (int j = 1; j < grid.ny - 1; ++j) {
            for (int i = 1; i < grid.nx - 1; ++i) {
                double dudx = (grid.u_star[grid.idx(i+1,j)] - grid.u_star[grid.idx(i-1,j)]) / (2*grid.dx);
                double dvdy = (grid.v_star[grid.idx(i,j+1)] - grid.v_star[grid.idx(i,j-1)]) / (2*grid.dy);
                grid.rhs[grid.idx(i,j)] = (dudx + dvdy) / dt;
            }
        }
        
        // Jacobi iteration
        for (int iter = 0; iter < max_iter; ++iter) {
            std::copy(grid.p.begin(), grid.p.end(), grid.p_old.begin());
            
            for (int j = 1; j < grid.ny - 1; ++j) {
                for (int i = 1; i < grid.nx - 1; ++i) {
                    grid.p[grid.idx(i,j)] = factor * (
                        (grid.p_old[grid.idx(i+1,j)] + grid.p_old[grid.idx(i-1,j)]) / dx2 +
                        (grid.p_old[grid.idx(i,j+1)] + grid.p_old[grid.idx(i,j-1)]) / dy2 -
                        grid.rhs[grid.idx(i,j)]
                    );
                }
            }
        }
    }
    
    void project_velocity() {
        for (int j = 1; j < grid.ny - 1; ++j) {
            for (int i = 1; i < grid.nx - 1; ++i) {
                double dpdx = (grid.p[grid.idx(i+1,j)] - grid.p[grid.idx(i-1,j)]) / (2*grid.dx);
                double dpdy = (grid.p[grid.idx(i,j+1)] - grid.p[grid.idx(i,j-1)]) / (2*grid.dy);
                grid.u[grid.idx(i,j)] = grid.u_star[grid.idx(i,j)] - dt * dpdx;
                grid.v[grid.idx(i,j)] = grid.v_star[grid.idx(i,j)] - dt * dpdy;
            }
        }
    }
    
    void apply_bc() {
        for (int j = 0; j < grid.ny; ++j) {
            grid.U(0, j) = 0; grid.V(0, j) = 0;
            grid.U(grid.nx-1, j) = 0; grid.V(grid.nx-1, j) = 0;
        }
        for (int i = 0; i < grid.nx; ++i) {
            grid.U(i, 0) = 0; grid.V(i, 0) = 0;
            grid.U(i, grid.ny-1) = 1.0; grid.V(i, grid.ny-1) = 0;
        }
    }
    
    void step() {
        compute_tentative_velocity();
        solve_pressure(100);
        project_velocity();
        apply_bc();
    }
    
    double run(int steps) {
        initialize();
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int s = 0; s < steps; ++s) {
            step();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        return elapsed.count();
    }
};


int main(int argc, char* argv[]) {
    std::cout << "==============================================\n";
    std::cout << "Navier-Stokes 2D Parallel Benchmark\n";
    std::cout << "==============================================\n";
    
#if HAS_WINDOWS_THREADS
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    int hw_threads = sysinfo.dwNumberOfProcessors;
    std::cout << "Using: Windows Native Threads\n";
    std::cout << "Hardware threads: " << hw_threads << "\n";
#else
    int hw_threads = 1;
    std::cout << "Threading: Not available (serial only)\n";
#endif
    
    // Parse args
    int grid_size = 64;
    int num_steps = 100;
    int max_threads = hw_threads > 0 ? hw_threads : 4;
    
    if (argc >= 2) grid_size = std::atoi(argv[1]);
    if (argc >= 3) num_steps = std::atoi(argv[2]);
    if (argc >= 4) max_threads = std::atoi(argv[3]);
    
    std::cout << "\nBenchmark: " << grid_size << "x" << grid_size 
              << ", " << num_steps << " steps\n";
    std::cout << "----------------------------------------------\n";
    
    double serial_time = 0.0;
    
    for (int threads = 1; threads <= max_threads; ++threads) {
        Solver solver(grid_size, grid_size, 100.0, threads);
        double time = solver.run(num_steps);
        
        if (threads == 1) {
            serial_time = time;
        }
        
        double speedup = serial_time / time;
        double efficiency = (speedup / threads) * 100.0;
        
        printf("Threads: %2d | Time: %7.3fs | Speedup: %5.2fx | Efficiency: %5.1f%%\n",
               threads, time, speedup, efficiency);
    }
    
    std::cout << "----------------------------------------------\n";
    std::cout << "Benchmark complete!\n";
    
    return 0;
}
