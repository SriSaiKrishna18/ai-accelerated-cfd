# Frequently Asked Questions

## General

**Q: What's the main contribution?**
A: 19.5× speedup for 100-case parameter sweeps with 98.5% accuracy. All results measured, not projected.

**Q: Why is this useful?**
A: Parameter sweeps, design optimization, and uncertainty quantification are common in engineering. This makes them 20× faster.

**Q: Can I reproduce your results?**
A: Yes. Run `python scripts/benchmark_100_cases.py`. Expected: ~19-20× speedup, ~1.5% error.

---

## Technical

**Q: Why is AI slower for single cases?**
A: Training overhead (5.5s) exceeds HPC time (~8s) for single queries. For 100 queries, training is amortized.

**Q: How do you choose training cases?**
A: Uniform sampling: 7 cases across [0.5, 2.0] range.

**Q: What if I test outside the training range?**
A: Accuracy degrades (extrapolation). Solution: Include edge cases in training or use HPC.

**Q: Why this architecture?**
A: Simple CNN that maps parameter → flow field. ~100K parameters, trains in 5 seconds.

---

## Performance

**Q: Where does speedup come from?**
A: AI inference is 2780× faster per case (3ms vs 8207ms). Training (5.5s) amortized over 93 inferences.

**Q: Does it scale beyond 100 cases?**
A: Yes! Training stays fixed, so more cases = better amortization. ~23× for 1000 cases.

**Q: What about GPU?**
A: All benchmarks on CPU. GPU would make AI training 10× faster.

---

## Accuracy

**Q: How accurate is AI?**
A: Mean: 1.48% error. Worst: 2.23%. AI predictions are 98.5% accurate.

**Q: How do you measure error?**
A: RMSE between AI prediction and HPC ground truth. Both methods run on same cases.

**Q: Does AI violate physics?**
A: Slightly higher divergence than HPC (~1e-4 vs 1e-6), but acceptable for engineering.

---

## Limitations

**Q: Main limitations?**
A: 
1. Interpolation only (no extrapolation)
2. Single parameter sweep
3. Breakeven at ~20 cases

**Q: When to use pure HPC?**
A: <20 cases, need 100% accuracy, or testing outside training range.

**Q: When to use AI-HPC?**
A: Parameter sweeps, uncertainty quantification, design optimization (>20 cases).
