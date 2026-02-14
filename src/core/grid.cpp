#include "grid.h"

Grid::Grid(int nx, int ny, double lx, double ly)
    : nx_(nx), ny_(ny), lx_(lx), ly_(ly) {
    // Compute grid spacing
    dx_ = lx / (nx - 1);
    dy_ = ly / (ny - 1);
}

void Grid::get_coords(int idx, int& i, int& j) const {
    i = idx % nx_;
    j = idx / nx_;
}
