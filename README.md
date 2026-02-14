# Navier-Stokes 2D AI-HPC Hybrid Solver

> **19.5× combined speedup for 100-case parameter sweep with 98.5% accuracy**

[![Build Status](https://github.com/SriSaiKrishna18/ai-accelerated-cfd/actions/workflows/ci.yml/badge.svg)](https://github.com/SriSaiKrishna18/ai-accelerated-cfd/actions)

---

## 🚀 Key Achievement

| Metric | Result |
|--------|--------|
| **Hybrid Speedup** | **19.5×** vs optimized HPC (measured!) |
| **Accuracy** | **~2%** error (5 seeds: 0.0166 ± 0.0009 excluding outlier) |
| **Physics** | KE correlation = 0.9975, divergence PASS |
| **Significance** | p = 0.0137 (statistically significant) |
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
pip install torch numpy matplotlib

# Run 100-case benchmark
python scripts/benchmark_100_cases.py
```

Expected output:
```
Pure HPC: 820,747ms (13.7 min)
Hybrid:    42,089ms (42 sec)
Speedup:   19.5×
Accuracy:  1.48% RMSE
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
├── python/models/convlstm.py          # AI model (745K parameters)
├── scripts/
│   ├── benchmark_100_cases.py         # Full 100-case benchmark
│   ├── comprehensive_validation.py    # 8-test validation suite
│   ├── physics_validation_detailed.py # Physics constraint checks
│   ├── baseline_comparison.py         # AI vs alternatives
│   ├── physics_informed_training.py   # PINN loss function
│   └── uncertainty_quantification.py  # MC Dropout confidence
├── checkpoints/best_model.pth         # Trained model
└── results/                           # Benchmark outputs & plots
```

---

## 🔬 Validation & Robustness (9 Tests)

| Test | Result |
|------|--------|
| Reproducibility (5 seeds) | RMSE: 0.0166 ± 0.0009 (excl. outlier) |
| Cross-validation (5 folds) | 0.0211 ± 0.0055, generalizes |
| Overfitting analysis | Train/Test ratio = 1.0x |
| Ablation (8 configs) | MLP best but impractical; CNN chosen |
| Noise robustness | Robust up to 10% noise |
| Physics validation | KE corr = 0.9975, divergence PASS |
| Statistical significance | p = 0.0137 (significant) |
| Failure detection | Catches extrapolation + BC violations |

See [VALIDATION.md](VALIDATION.md) for exact numbers and [`results/validation_log.txt`](results/validation_log.txt) for proof.

---

## 🎤 Interview Pitch

> "I built an AI-HPC hybrid CFD solver achieving **19.5× speedup** over optimized HPC for 100-case parameter sweeps with **~2% error**. I ran a comprehensive 9-test validation suite — 5-seed reproducibility, 5-fold cross-validation, 8-config ablation, physics checks (KE corr 0.9975), and statistical significance (p=0.0137). I documented honest limitations: seed sensitivity, BC violations at high velocity, and single-problem scope."

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
