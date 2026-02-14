#!/usr/bin/env python3
"""
GPU Hybrid Timing Script for Kaggle/Colab

This script demonstrates that AI inference on GPU is faster than HPC simulation.
Run on Kaggle with GPU for accurate timing.

Expected Results:
- AI inference on GPU: ~10-50ms per prediction
- HPC simulation: ~100-500ms for equivalent time steps
- Hybrid speedup: 2-5× faster than pure HPC
"""

import time
import numpy as np
import subprocess
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def time_hpc_simulation(grid_size, num_steps):
    """Time HPC simulation for a given number of steps"""
    executable = './build/ns_optimized.exe'
    
    if not os.path.exists(executable.replace('./', '')):
        print(f"Building {executable}...")
        os.system('g++ -std=c++17 -O3 -march=native src/optimized_solver.cpp -o build/ns_optimized.exe')
    
    cmd = [executable, str(grid_size), str(num_steps), '1']
    
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed_ms = (time.time() - start) * 1000
        
        # Parse actual solver time from output
        for line in result.stdout.split('\n'):
            if 'Time:' in line and 'ms' in line:
                parts = line.split('Time:')[-1].strip()
                elapsed_ms = float(parts.replace('ms', '').strip())
                break
        
        return elapsed_ms
    except Exception as e:
        print(f"Error: {e}")
        return None


def time_ai_prediction_gpu(model, input_tensor, num_future_steps, device):
    """Time AI prediction on GPU"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor, future_steps=5)
    
    # Synchronize for accurate GPU timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        predictions = model(input_tensor, future_steps=num_future_steps)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_ms = (time.time() - start) * 1000
    return elapsed_ms, predictions


def run_hybrid_timing_demo():
    """Main timing demonstration"""
    
    print("=" * 70)
    print("HYBRID HPC+AI TIMING DEMONSTRATION")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load model
    from models.convlstm import ConvLSTM
    
    model = ConvLSTM(input_dim=3, hidden_dims=[64, 64, 64])
    
    model_path = 'checkpoints/best_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print("Warning: Using untrained model (no checkpoint found)")
    
    model = model.to(device)
    
    # Scenarios
    scenarios = [
        {'grid': 64, 'hpc_steps': 50, 'ai_steps': 50},
        {'grid': 128, 'hpc_steps': 50, 'ai_steps': 50},
        {'grid': 256, 'hpc_steps': 20, 'ai_steps': 20},
    ]
    
    results = []
    
    print("\n" + "-" * 70)
    
    for scenario in scenarios:
        grid = scenario['grid']
        hpc_steps = scenario['hpc_steps']
        ai_steps = scenario['ai_steps']
        
        print(f"\n[Grid: {grid}×{grid}]")
        
        # Method 1: Full HPC simulation
        print(f"  HPC ({hpc_steps + ai_steps} steps)... ", end='', flush=True)
        hpc_full_time = time_hpc_simulation(grid, hpc_steps + ai_steps)
        print(f"{hpc_full_time:.1f} ms")
        
        # Method 2: Hybrid (HPC + AI)
        print(f"  HPC ({hpc_steps} steps)... ", end='', flush=True)
        hpc_checkpoint_time = time_hpc_simulation(grid, hpc_steps)
        print(f"{hpc_checkpoint_time:.1f} ms")
        
        # AI prediction
        print(f"  AI ({ai_steps} future steps)... ", end='', flush=True)
        input_tensor = torch.randn(1, 3, grid, grid).to(device)
        ai_time, _ = time_ai_prediction_gpu(model, input_tensor, ai_steps, device)
        print(f"{ai_time:.1f} ms")
        
        # Calculate hybrid total
        hybrid_total = hpc_checkpoint_time + ai_time
        speedup = hpc_full_time / hybrid_total if hybrid_total > 0 else 0
        time_saved_pct = (1 - hybrid_total / hpc_full_time) * 100 if hpc_full_time > 0 else 0
        
        print(f"\n  Results:")
        print(f"    Full HPC:     {hpc_full_time:.1f} ms")
        print(f"    Hybrid:       {hybrid_total:.1f} ms ({hpc_checkpoint_time:.1f} + {ai_time:.1f})")
        print(f"    Speedup:      {speedup:.2f}×")
        print(f"    Time saved:   {time_saved_pct:.1f}%")
        
        results.append({
            'grid': grid,
            'hpc_full': hpc_full_time,
            'hpc_checkpoint': hpc_checkpoint_time,
            'ai_time': ai_time,
            'hybrid_total': hybrid_total,
            'speedup': speedup,
            'time_saved_pct': time_saved_pct
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Grid':>10} {'Full HPC':>12} {'Hybrid':>12} {'Speedup':>10} {'Saved':>10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['grid']:>10} {r['hpc_full']:>10.1f}ms {r['hybrid_total']:>10.1f}ms "
              f"{r['speedup']:>9.2f}× {r['time_saved_pct']:>9.1f}%")
    
    print("=" * 70)
    
    # Compute average
    avg_speedup = np.mean([r['speedup'] for r in results if r['speedup'] > 0])
    print(f"\nAverage Speedup: {avg_speedup:.2f}×")
    
    if avg_speedup > 1:
        print("✓ HYBRID APPROACH IS FASTER THAN PURE HPC!")
    else:
        print("✗ Need GPU for speedup (CPU AI inference is slow)")
    
    return results


def create_timing_plot(results):
    """Create visualization of timing results"""
    import matplotlib.pyplot as plt
    
    grids = [f"{r['grid']}×{r['grid']}" for r in results]
    hpc_times = [r['hpc_full'] for r in results]
    hybrid_times = [r['hybrid_total'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    x = np.arange(len(grids))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, hpc_times, width, label='Full HPC', color='steelblue')
    bars2 = ax1.bar(x + width/2, hybrid_times, width, label='Hybrid (HPC+AI)', color='orange')
    
    ax1.set_ylabel('Time (milliseconds)', fontsize=12)
    ax1.set_xlabel('Grid Size', fontsize=12)
    ax1.set_title('Runtime Comparison: HPC vs Hybrid', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(grids)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Speedup chart
    speedups = [r['speedup'] for r in results]
    bars = ax2.bar(grids, speedups, color='green', alpha=0.7, edgecolor='black')
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='No speedup')
    ax2.set_ylabel('Speedup (×)', fontsize=12)
    ax2.set_xlabel('Grid Size', fontsize=12)
    ax2.set_title('Hybrid Speedup vs Full HPC', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}×', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/hybrid_gpu_speedup.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved: results/hybrid_gpu_speedup.png")
    plt.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'python')
    
    os.makedirs('results', exist_ok=True)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Please install: pip install torch")
        exit(1)
    
    results = run_hybrid_timing_demo()
    
    try:
        create_timing_plot(results)
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    print("\nDone!")
