# Reproducibility Guide

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib
- C++ compiler with OpenMP (MinGW-w64 or GCC)

---

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install torch numpy matplotlib
```

### 2. Run Benchmark
```bash
python scripts/benchmark_100_cases.py
```

**Expected output:**
```
Pure HPC: 820,747ms (13.7 min)
Hybrid:    42,089ms (42 sec)
Speedup:   19.5×
Accuracy:  1.48% RMSE
```

---

## Detailed Reproduction

### Reproduce Multi-Query Results

```bash
# Full 100-case test (~15 minutes)
python scripts/benchmark_100_cases.py

# Quick 20-case test (~3 minutes)
python scripts/multi_query_benchmark.py
```

### Reproduce HPC Optimization

```bash
# Build with OpenMP (Windows)
$env:PATH = "C:\mingw64\bin;" + $env:PATH
g++ -std=c++17 -O3 -fopenmp src/optimized_solver.cpp -o build/ns_omp.exe

# Run benchmark
./build/ns_omp.exe 256 50
```

---

## Expected Results

| Metric | Expected Value | Acceptable Range |
|--------|----------------|------------------|
| Speedup | 19.5× | 18-21× |
| Mean RMSE | 0.0148 | 0.012-0.018 |
| Max RMSE | 0.0223 | <0.025 |
| Training loss | 0.0002 | <0.001 |

---

## Troubleshooting

**"PyTorch not found"**
```bash
pip install torch
```

**"Speedup lower than 18×"**
Normal variation. Check HPC solver converged properly.

**"RMSE higher than 2.5%"**
Retrain model or check training data quality.

---

## Files Required

```
AI_HPC/
├── scripts/benchmark_100_cases.py  # Main benchmark
├── scripts/multi_query_benchmark.py
├── requirements.txt
└── python/                         # Supporting modules
```
