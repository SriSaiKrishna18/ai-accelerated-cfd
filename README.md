# 2D Pressure-Velocity Flow Surrogate Model

> **AI-accelerated parameter sweeps: 19.5× speedup with ~2% error**

[![Build Status](https://github.com/SriSaiKrishna18/ai-accelerated-cfd/actions/workflows/ci.yml/badge.svg)](https://github.com/SriSaiKrishna18/ai-accelerated-cfd/actions)

---

## ⚠️ Solver Disclaimer

This project uses a **simplified 2D pressure-velocity model** with explicit Euler time integration.
It is **NOT** a full incompressible Navier-Stokes solver:
- Divergence: ~22 (a proper incompressible solver achieves ~1e-6)
- Pressure Poisson equation is not iterated to convergence
- Results demonstrate **relative accuracy** (AI vs this solver), not absolute physics fidelity

For production CFD, use established solvers (OpenFOAM, Fluent, SU2).

**The AI/HPC integration methodology, speedup measurement, and validation approach are all real and transferable to proper solvers.**

---

## 🚀 Key Achievement

| Metric | Result |
|--------|--------|
| **Hybrid Speedup** | **19.5×** vs optimized HPC (measured) |
| **Accuracy** | ~2% RMSE (relative to this solver) |
| **Physics** | KE correlation > 0.99, BC enforced post-hoc |
| **Validation** | 9 tests, see [VALIDATION.md](VALIDATION.md) |

```
How 19.5× is measured:
  Optimized HPC (100 cases):  100 × 8.2s = 820s
  AI-HPC Hybrid (100 cases):  7×8.2s HPC + 5.5s train + 0.27s infer = 42s
  Speedup: 820s / 42s = 19.5×

  This is AI hybrid vs OPTIMIZED HPC (with OpenMP + Red-Black GS)
  NOT vs unoptimized baseline.
```

**All results measured. Proof: [`results/validation_log.txt`](results/validation_log.txt)**

---

## 💡 The Key Insight

**Single simulation**: AI is slower (training overhead)
**Multi-query (100+ cases)**: AI provides **19.5× speedup**

Why? Training is ONE-TIME, inference is 2780× faster per case.

---

## Results

### Multi-Query Speedup (100 Cases)

| Method | Time | Cases | Speedup |
|--------|------|-------|---------|
| Pure HPC | 13.7 min | 100 | baseline |
| **AI-HPC Hybrid** | **42 sec** | **100** | **19.5×** |

### Breakdown

| Step | Time | Cases |
|------|------|-------|
| HPC training | 36.3 sec | 7 |
| AI training | 5.5 sec | - |
| **AI inference** | **0.27 sec** | **93** |

### Accuracy (93 Test Cases)

| Metric | Value |
|--------|-------|
| Mean RMSE | 0.0148 (1.48% error) |
| Min RMSE | 0.0078 (0.78% error) |
| Max RMSE | 0.0223 (2.23% error) |
| Training loss | 0.000208 |

**All 93 test cases validated against HPC ground truth.**

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib scikit-learn

# Run 100-case benchmark
python scripts/benchmark_100_cases.py

# Run full validation suite (9 tests)
python scripts/comprehensive_validation.py
```

---

## 📊 Use Cases

| Scenario | Cases | HPC Time | Hybrid Time | Speedup |
|----------|-------|----------|-------------|---------|
| Parameter sweep | 100 | 13.7 min | 42 sec | **19.5×** |
| Design optimization | 500 | 68 min | 3.5 min | **~19×** |
| Monte Carlo UQ | 1000 | 137 min | 6 min | **~23×** |

---

## 📁 Project Structure

```
AI_HPC/
├── src/optimized_solver.cpp           # HPC solver (OpenMP + Red-Black GS)
├── python/models/convlstm.py          # AI model architecture
├── scripts/
│   ├── benchmark_100_cases.py         # Full 100-case benchmark
│   ├── comprehensive_validation.py    # 9-test validation suite (v2)
│   ├── physics_validation_detailed.py # Physics constraint checks
│   ├── baseline_comparison.py         # AI vs alternatives
│   ├── physics_informed_training.py   # PINN loss function
│   └── uncertainty_quantification.py  # MC Dropout confidence
├── checkpoints/best_model.pth         # Trained model
└── results/                           # Benchmark outputs & plots
```

**Model:** CNN with 403K parameters (64-channel decoder).
Earlier experiments used ConvLSTM (745K params); the simpler CNN achieves similar accuracy with fewer parameters.

---

## 🔬 Validation & Robustness (9 Tests)

| Test | Result |
|------|--------|
| Reproducibility (5 seeds) | With early stopping, consistent RMSE |
| Cross-validation (5 folds) | v ∈ [0.1, 1.0] — consistent with training |
| Overfitting analysis | Train/test gap documented |
| Ablation (8 configs) | Includes train AND inference timing |
| Noise robustness | Robust up to 10% noise |
| Physics validation | KE correlation, BC enforcement (before/after) |
| Statistical significance | AI vs linear interp vs GP (with p-values) |
| Failure detection | Out-of-range detection, BC checks |

See [VALIDATION.md](VALIDATION.md) for exact numbers and [`results/validation_log.txt`](results/validation_log.txt) for proof.

---

## ⚠️ Known Limitations

1. **Not true Navier-Stokes**: Solver uses simplified Euler method (divergence ~22)
2. **Circular validation**: AI accuracy is relative to this solver, not true physics
3. **Boundary conditions**: Enforced post-hoc, not physics-informed
4. **Single problem**: Only 2D lid-driven cavity tested
5. **No SOTA comparison**: Not compared to POD, FNO, or published methods

---

## 🎤 Interview Pitch

> "I built an AI-HPC hybrid surrogate model achieving **19.5× speedup** for 100-case parameter sweeps with ~2% error relative to the baseline solver. I ran 9 validation tests with exact numbers — 5-seed reproducibility with early stopping, 5-fold cross-validation on [0.1, 1.0], ablation with inference timing, and statistical significance (AI vs linear interpolation vs GP regression). I document honest limitations: the solver is simplified (not full NS), accuracy is relative, and BCs are enforced post-hoc. The methodology transfers to production solvers."

---

## Documentation

- [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) - One-page overview
- [RESULTS.md](RESULTS.md) - Complete results documentation
- [VALIDATION.md](VALIDATION.md) - Validation & robustness testing
- [FAQ.md](FAQ.md) - Frequently asked questions
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) - How to reproduce results
- [LIMITATIONS.md](LIMITATIONS.md) - Known limitations

---

## License

MIT
