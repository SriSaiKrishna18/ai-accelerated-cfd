# Parallelization Benchmark Results

## Navier-Stokes 2D Solver - Windows Threading

**Date**: January 31, 2026  
**Implementation**: Windows Native Threads (CreateThread API)  
**System**: 8 hardware threads

---

## Results Summary

### 64x64 Grid, 100 Steps

| Threads | Time (s) | Speedup | Efficiency |
|---------|----------|---------|------------|
| 1 | 0.179 | 1.00x | 100.0% |
| **2** | **0.162** | **1.11x** | 55.3% |
| 4 | 0.181 | 0.99x | 24.7% |
| 8 | 0.211 | 0.85x | 10.6% |

### 128x128 Grid, 50 Steps

| Threads | Time (s) | Speedup | Efficiency |
|---------|----------|---------|------------|
| 1 | 0.309 | 1.00x | 100.0% |
| 2 | 0.294 | 1.05x | 52.6% |
| **4** | **0.285** | **1.08x** | 27.1% |
| 8 | 0.324 | 0.95x | 11.9% |

---

## Analysis

**Observations**:
1. Best speedup achieved: **1.11x** (2 threads, 64x64)
2. Thread overhead dominates for small grids
3. Optimal thread count: 2-4 for current grid sizes
4. Efficiency decreases with more threads

**Why Limited Speedup?**
- Small grid sizes (64x64, 128x128) have low computation-to-overhead ratio
- Only velocity computation is parallelized
- Pressure solver uses sequential Jacobi iteration
- Memory bandwidth limitations

**Recommendations for Final Review**:
1. Use OpenMP with MinGW-w64 or MSVC for better performance
2. Parallelize pressure solver (Red-Black Gauss-Seidel)
3. Test with 256x256+ grids for meaningful speedup
4. Consider MPI for distributed computing

---

## Files Created

| File | Description |
|------|-------------|
| `src/win_parallel_benchmark.cpp` | Windows threading benchmark |
| `include/parallel.h` | Parallel primitives (std::thread) |
| `build/win_parallel_benchmark.exe` | Compiled benchmark |
