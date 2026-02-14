#ifndef GRID_H
#define GRID_H

#include <vector>
#include <string>

/**
 * @class Grid
 * @brief Represents a 2D structured grid for CFD simulations.
 * 
 * Handles spatial discretization with uniform grid spacing.
 * Uses column-major indexing: index(i, j) = i + j * nx
 */
class Grid {
public:
    /**
     * @brief Construct a new Grid object
     * @param nx Number of grid points in x-direction
     * @param ny Number of grid points in y-direction
     * @param lx Physical length in x-direction
     * @param ly Physical length in y-direction
     */
    Grid(int nx, int ny, double lx, double ly);
    
    // Accessors
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    double lx() const { return lx_; }
    double ly() const { return ly_; }
    
    /**
     * @brief Convert 2D indices to 1D array index
     * @param i x-index (0 to nx-1)
     * @param j y-index (0 to ny-1)
     * @return 1D array index
     */
    int index(int i, int j) const { return i + j * nx_; }
    
    /**
     * @brief Get total number of grid points
     */
    int size() const { return nx_ * ny_; }
    
    /**
     * @brief Convert 1D index back to 2D coordinates
     */
    void get_coords(int idx, int& i, int& j) const;
    
    /**
     * @brief Get x-coordinate at grid point i
     */
    double x(int i) const { return i * dx_; }
    
    /**
     * @brief Get y-coordinate at grid point j
     */
    double y(int j) const { return j * dy_; }
    
private:
    int nx_, ny_;       ///< Number of grid points
    double lx_, ly_;    ///< Physical domain size
    double dx_, dy_;    ///< Grid spacing
};

#endif // GRID_H
