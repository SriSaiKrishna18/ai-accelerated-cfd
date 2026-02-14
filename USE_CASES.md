# Real-World Use Cases for AI-HPC Hybrid CFD

## Overview

This document provides concrete examples where our AI-HPC hybrid approach provides real computational benefit.

---

## Use Case 1: Aerospace Design Optimization

### Scenario
An aerospace engineer is designing a wing flap and needs to test 500 different flap angles to find the optimal configuration.

### Comparison

| Approach | Simulations | Time per sim | Total Time |
|----------|-------------|--------------|------------|
| Pure HPC | 500 | 100ms | 50,000ms (50s) |
| AI-HPC Hybrid | 7 HPC + 493 AI | 700ms + 2,465ms | 3,165ms (3.2s) |

**Speedup: 15.8×**

### Impact
- Enables rapid iteration during a single design meeting
- 50 seconds → 3.2 seconds = real-time feedback

---

## Use Case 2: Uncertainty Quantification

### Scenario
A nuclear engineer needs to understand how manufacturing tolerances affect cooling flow in a reactor component. This requires Monte Carlo simulation with 1,000 perturbed initial conditions.

### Comparison

| Approach | Simulations | Total Time |
|----------|-------------|------------|
| Pure HPC | 1,000 | 100,000ms (100s) |
| AI-HPC Hybrid | 10 HPC + 990 AI | 6,165ms (6.2s) |

**Speedup: 16.2×**

### Impact
- What took 1.5 minutes now takes 6 seconds
- Enables uncertainty bounds in design decisions
- Critical for safety-critical applications

---

## Use Case 3: Reynolds Number Parameter Sweep

### Scenario
A researcher studying turbulence transition needs to identify the critical Reynolds number by testing Re = 100, 200, 300, ..., 10,000 (100 values).

### Comparison

| Approach | Simulations | Total Time |
|----------|-------------|------------|
| Pure HPC | 100 | 10,000ms |
| AI-HPC Hybrid | 7 HPC + 93 AI | 1,665ms |

**Speedup: 6.0×**

### Impact
- PhD students can explore parameter space quickly
- More thorough studies in same time budget

---

## When to Use AI-HPC Hybrid

### ✅ Use Hybrid When:
- Running 20+ similar simulations
- Parameter sweeps
- Monte Carlo / uncertainty quantification
- Design optimization loops
- Sensitivity analysis

### ❌ Use Pure HPC When:
- Single simulation needed
- Highly turbulent flows (AI may struggle)
- Extrapolation far beyond training data
- Highest possible accuracy required

---

## Cost Analysis

### Assumptions
- Cloud HPC: $0.10 per CPU-hour
- 1000 simulations × 100ms = 100s = $0.003 HPC cost

### Savings
| Scenario | HPC Cost | Hybrid Cost | Savings |
|----------|----------|-------------|---------|
| 1000 sims/day | $0.003 | $0.0002 | 93% |
| 100,000 sims/month | $0.30 | $0.02 | 93% |

*Primary value is TIME, not cost (simulations are fast enough that cost is minimal)*

---

## Conclusion

AI-HPC hybrid provides **6-16× speedup** for multi-query scenarios that are standard in engineering practice. The key is amortizing training cost over many inferences.
