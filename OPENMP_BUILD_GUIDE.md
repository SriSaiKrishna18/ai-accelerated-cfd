# OpenMP Build Guide

## Current Issue
Your MinGW (6.3.0) doesn't have pthread support needed for OpenMP.

## Solutions

### Option 1: MinGW-w64 (Recommended)
Download and install MinGW-w64 with POSIX threads:
1. Go to: https://sourceforge.net/projects/mingw-w64/
2. Choose: x86_64-posix-seh
3. Add to PATH
4. Build with:
   ```bash
   g++ -std=c++17 -O2 -fopenmp -DUSE_OPENMP -I include src/core/*.cpp src/main.cpp -o build/ns_main_omp.exe
   ```

### Option 2: Visual Studio (MSVC)
1. Install Visual Studio with C++ workload
2. Open Developer Command Prompt
3. Build with:
   ```bash
   cl /O2 /openmp /DUSE_OPENMP /I include src\core\grid.cpp src\core\solver.cpp src\main.cpp /Fe:build\ns_main_omp.exe
   ```

### Option 3: WSL (Windows Subsystem for Linux)
1. Enable WSL2
2. Install Ubuntu from Microsoft Store
3. Install GCC: `sudo apt install g++`
4. Build with:
   ```bash
   g++ -std=c++17 -O2 -fopenmp -DUSE_OPENMP -I include src/core/*.cpp src/main.cpp -o build/ns_main_omp
   ```

### Option 4: Use GitHub Actions (CI/CD)
Create `.github/workflows/build.yml` to build on Linux/Mac runners.

## Verification
After building with OpenMP:
```bash
# Set thread count
set OMP_NUM_THREADS=4

# Run benchmark
python scripts/benchmark_parallel.py
```

## OpenMP Code Already Added
The solver already has OpenMP pragmas:
- `compute_tentative_velocity()` - collapse(2) parallel for
- `solve_pressure_poisson()` - parallel RHS + parallel Jacobi with reduction
- `project_velocity()` - collapse(2) parallel for

Just compile with `-fopenmp -DUSE_OPENMP` to enable.
