# Reproducibility Guide

## System Requirements

- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.8+
- **C++ Compiler**: MinGW-w64 8.1+ (Windows) or GCC 8+ (Linux)
- **RAM**: 8GB minimum
- **GPU**: Optional (for faster AI training)

---

## Step 1: Environment Setup

### Python Dependencies
```bash
pip install -r requirements.txt
```

### C++ Compiler (Windows)
```powershell
# Add MinGW to PATH
$env:PATH = "C:\mingw64\bin;" + $env:PATH

# Verify
g++ --version
```

---

## Step 2: Build HPC Solver

```bash
# Create build directory
mkdir -p build

# Compile with OpenMP
g++ -std=c++17 -O3 -fopenmp src/optimized_solver.cpp -o build/ns_omp.exe
```

---

## Step 3: Generate Training Data

```bash
# Generate lid-driven cavity simulation data
python python/generate_training_data.py --grid_size 64 --num_timesteps 1000
```

This creates:
- `data/simulation_64x64.npz` - Raw simulation data
- Folder: `data/` with checkpoint files

---

## Step 4: Train AI Model

```bash
# Train ConvLSTM model
python scripts/ai_accelerated_solver.py
```

Output:
- `checkpoints/best_model.pth` - Trained model weights
- Training logs in console

---

## Step 5: Run Multi-Query Benchmark

```bash
# Demonstrates AI-HPC speedup
python scripts/multi_query_benchmark.py
```

Expected output:
- 6× speedup at 100 queries
- 16× speedup at 1000 queries
- Graph: `results/multi_query_speedup.png`

---

## Step 6: Validate Results

```bash
# Physics validation
python scripts/physics_validation.py

# Accuracy validation
python python/visualize.py
```

---

## Expected Results

| Metric | Value |
|--------|-------|
| AI accuracy (RMSE) | 0.45% |
| HPC speedup | 2.6× |
| Multi-query (100 cases) | 6× |
| Multi-query (1000 cases) | 16× |

---

## Troubleshooting

### "g++ not found"
Ensure MinGW is in PATH:
```powershell
$env:PATH = "C:\mingw64\bin;" + $env:PATH
```

### "No module named torch"
```bash
pip install torch
```

### "CUDA not available"
CPU training works but is slower. For GPU:
- Use Kaggle/Colab notebooks
- Or install CUDA toolkit + PyTorch with CUDA

---

## File Structure

```
AI_HPC/
├── src/optimized_solver.cpp    # HPC solver (2.6× speedup)
├── python/
│   ├── models/convlstm.py      # AI model architecture
│   └── visualize.py            # Visualization tools
├── scripts/
│   └── multi_query_benchmark.py # Main demo
├── checkpoints/best_model.pth   # Trained model
├── data/                        # Training data
├── results/                     # Output graphs
└── requirements.txt             # Dependencies
```

---

## Reproducing Key Results

### Result 1: Multi-Query Speedup
```bash
python scripts/multi_query_benchmark.py
```
Screenshot/save `results/multi_query_speedup.png`

### Result 2: AI Accuracy
```bash
python python/visualize.py
```
Shows 0.45% RMSE

### Result 3: HPC Speedup
```bash
build/ns_omp.exe
```
Shows 2.6× speedup (algorithmic + parallel)
