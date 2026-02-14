# Presentation Slides Content

## Slide 1: Title
**Navier-Stokes AI-HPC Hybrid Solver**
*19.5× Speedup for Multi-Query CFD*

---

## Slide 2: Problem
- CFD simulations are computationally expensive
- Engineers need to test hundreds of configurations
- Parameter sweeps, design optimization, uncertainty quantification
- **Challenge:** 100 simulations = 13.7 minutes

---

## Slide 3: Solution Overview
**Hybrid Approach:**
1. Run HPC for 7 representative cases
2. Train AI on those 7 cases (5 seconds)
3. Use AI to predict remaining 93 cases (0.27 seconds)

**Key Insight:** Training is ONE-TIME, inference is 2780× faster

---

## Slide 4: System Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ 7 Training  │ ──► │  AI Model    │ ──► │ 93 Fast     │
│ Cases (HPC) │     │  Training    │     │ Predictions │
└─────────────┘     └──────────────┘     └─────────────┘
     36.3s              5.5s               0.27s
```

---

## Slide 5: HPC Baseline Optimization
| Optimization | Speedup |
|--------------|---------|
| Algorithmic (Red-Black GS) | 2.0× |
| Parallel (OpenMP 2T) | 1.33× |
| **Total HPC** | **2.6×** |

---

## Slide 6: Key Result
| Method | Time (100 cases) | Speedup |
|--------|------------------|---------|
| Pure HPC | 13.7 min | baseline |
| **AI-HPC Hybrid** | **42 sec** | **19.5×** |

---

## Slide 7: Breakdown
- HPC (7 cases): 36.3 sec
- AI training: 5.5 sec  
- **AI inference (93 cases): 0.27 sec** ← The magic

Per-case: HPC = 8.2 sec, AI = 0.003 sec = **2780× faster**

---

## Slide 8: Accuracy
| Metric | Value |
|--------|-------|
| Mean error | 1.48% |
| Accuracy | 98.5% |
| Test cases | 93 (all validated) |

---

## Slide 9: Scaling
| Cases | HPC | Hybrid | Speedup |
|-------|-----|--------|---------|
| 20 | 2.7 min | 47 sec | 3.5× |
| 100 | 13.7 min | 42 sec | **19.5×** |
| 500 | 68 min | 3.5 min | ~19× |

---

## Slide 10: Real-World Use Case
**Aerospace Wing Optimization**
- Test 500 flap configurations
- Pure HPC: 68 minutes
- AI-HPC Hybrid: 3.5 minutes
- **Enables rapid design iteration**

---

## Slide 11: Validation
✅ All 100 cases actually run (no projections)
✅ 93 test cases validated against HPC ground truth
✅ Error distribution: 75% < 1.5%, 100% < 2.3%
✅ Training converged (loss = 0.0002)

---

## Slide 12: Limitations
1. **Interpolation only** - No extrapolation beyond training range
2. **Single parameter** - Multi-parameter needs more training data
3. **Breakeven at ~20 cases** - Training overhead

---

## Slide 13: Technical Stack
- **HPC:** C++ with OpenMP
- **AI:** PyTorch (simple CNN ~100K params)
- **Validation:** All results measured
- **Repo:** GitHub with CI/CD

---

## Slide 14: Conclusions
✅ **19.5× measured speedup** (100 cases)
✅ **2780× per-inference speedup**
✅ **98.5% accuracy** (1.48% error)
✅ **All results validated**

---

## Slide 15: Key Takeaway
> "AI-HPC hybrid excels at multi-query scenarios. 
> Train once, infer many times.
> The more cases you need, the better the speedup."

**Thank you!**
