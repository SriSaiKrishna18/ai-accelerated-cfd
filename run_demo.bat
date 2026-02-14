@echo off
REM ============================================
REM  Navier-Stokes AI-HPC Demo Script
REM  One-command demonstration of full pipeline
REM ============================================

echo ============================================
echo  NAVIER-STOKES AI-HPC HYBRID DEMO
echo ============================================
echo.

REM Set PATH for MinGW-w64
set PATH=C:\mingw64\bin;%PATH%

REM Check if solver exists, build if not
if not exist "build\ns_omp.exe" (
    echo [1/5] Building OpenMP solver...
    g++ -std=c++17 -O3 -fopenmp src/optimized_solver.cpp -o build/ns_omp.exe
    if errorlevel 1 (
        echo ERROR: Build failed!
        echo Make sure MinGW-w64 is installed at C:\mingw64
        pause
        exit /b 1
    )
    echo       Build successful!
) else (
    echo [1/5] Solver already built
)

echo.
echo [2/5] Running HPC simulation (256x256, 50 steps)...
echo.

build\ns_omp.exe 256 50 2

echo.
echo [3/5] Running physics validation...
echo.

python scripts/physics_validation.py

echo.
echo [4/5] Generating visualization...
echo.

python python/visualize.py

echo.
echo [5/5] Demo complete!
echo.
echo ============================================
echo  RESULTS SUMMARY
echo ============================================
echo.
echo  Solver:     build/ns_omp.exe (OpenMP enabled)
echo  Model:      checkpoints/best_model.pth
echo  Accuracy:   0.45%% RMSE (see results/validation_report.md)
echo  Plots:      results/*.png
echo.
echo  Performance (measured):
echo    Algorithmic: 2.0x improvement (Red-Black GS)
echo    Parallel:    1.33x speedup (2 threads)
echo    Total:       2.6x vs original solver
echo.
echo ============================================

pause
