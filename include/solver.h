#ifndef SOLVER_H
#define SOLVER_H

#include "grid.h"
#include <vector>
#include <string>
#include <memory>

/**
 * @class NavierStokesSolver
 * @brief 2D incompressible Navier-Stokes solver using finite differences.
 * 
 * Implements the projection method:
 * 1. Compute tentative velocity (advection + diffusion)
 * 2. Solve pressure Poisson equation
 * 3. Project velocity to divergence-free field
 * 
 * Supports multiple initial conditions and boundary conditions.
 */
class NavierStokesSolver {
public:
    /**
     * @struct Parameters
     * @brief Simulation parameters
     */
    struct Parameters {
        int nx = 128;           ///< Grid points in x
        int ny = 128;           ///< Grid points in y
        double lx = 1.0;        ///< Domain length x
        double ly = 1.0;        ///< Domain length y
        double dt = 0.001;      ///< Time step
        double nu = 0.01;       ///< Kinematic viscosity
        double rho = 1.0;       ///< Density
        int max_iter_pressure = 1000;  ///< Max pressure solver iterations
        double tolerance = 1e-6;        ///< Pressure solver tolerance
        std::string bc_type = "lid_driven_cavity";  ///< Boundary condition type
        std::string ic_type = "lid_driven_cavity";  ///< Initial condition type
    };
    
    /**
     * @struct State
     * @brief Current simulation state
     */
    struct State {
        std::vector<double> u;  ///< x-velocity field
        std::vector<double> v;  ///< y-velocity field
        std::vector<double> p;  ///< Pressure field
        double time = 0.0;      ///< Current simulation time
        int step = 0;           ///< Current time step number
    };
    
    /**
     * @brief Construct solver with given parameters
     */
    explicit NavierStokesSolver(const Parameters& params);
    
    virtual ~NavierStokesSolver() = default;
    
    // ===== Main simulation methods =====
    
    /**
     * @brief Initialize fields based on ic_type
     */
    void initialize();
    
    /**
     * @brief Advance one time step
     */
    virtual void step();
    
    /**
     * @brief Run simulation until time t_final
     */
    void solve_until(double t_final);
    
    // ===== I/O methods =====
    
    /**
     * @brief Save state to binary checkpoint file
     */
    void save_checkpoint(const std::string& filename) const;
    
    /**
     * @brief Load state from binary checkpoint file
     */
    void load_checkpoint(const std::string& filename);
    
    /**
     * @brief Export state to VTK format for visualization
     */
    void save_vtk(const std::string& filename) const;
    
    // ===== Accessors =====
    
    const State& state() const { return state_; }
    const Grid& grid() const { return *grid_; }
    const Parameters& params() const { return params_; }
    
    // ===== Diagnostics =====
    
    /**
     * @brief Compute maximum divergence (should be ~0)
     */
    double compute_divergence() const;
    
    /**
     * @brief Compute total kinetic energy
     */
    double compute_kinetic_energy() const;
    
    /**
     * @brief Compute CFL number (should be < 1 for stability)
     */
    double compute_cfl_number() const;
    
protected:
    Parameters params_;
    std::unique_ptr<Grid> grid_;
    State state_;
    
    // Work arrays
    std::vector<double> u_star_;  ///< Tentative x-velocity
    std::vector<double> v_star_;  ///< Tentative y-velocity
    std::vector<double> rhs_;     ///< Pressure Poisson RHS
    
    // ===== Core numerical methods =====
    
    /**
     * @brief Compute tentative velocity (ignoring pressure)
     */
    virtual void compute_tentative_velocity();
    
    /**
     * @brief Solve pressure Poisson equation using Jacobi iteration
     */
    virtual void solve_pressure_poisson();
    
    /**
     * @brief Project velocity to divergence-free field
     */
    virtual void project_velocity();
    
    /**
     * @brief Apply boundary conditions to all fields
     */
    void apply_boundary_conditions();
    
    // ===== Initial conditions =====
    
    void set_lid_driven_cavity();
    void set_taylor_green();
    void set_shear_layer();
    void set_vortex_pair();
};

#endif // SOLVER_H
