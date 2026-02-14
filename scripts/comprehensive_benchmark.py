#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for Navier-Stokes Solver
Generates performance analysis plots and CSV data
"""

import subprocess
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv

def run_benchmark(executable, grid_size, num_threads, time_final=0.5, num_steps=None):
    """Run single benchmark and return time in seconds"""
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(num_threads)
    
    if num_steps:
        cmd = [executable, str(grid_size), str(num_steps), str(num_threads)]
    else:
        cmd = [executable, str(grid_size), str(num_steps or 50), str(num_threads)]
    
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              env=env, timeout=300)
        elapsed = time.time() - start
        
        # Try to parse output for more accurate timing
        for line in result.stdout.split('\n'):
            if 'Time:' in line or 'Runtime:' in line:
                parts = line.split(':')[-1].strip()
                if 'ms' in parts:
                    return float(parts.replace('ms', '').strip()) / 1000.0
                elif 's' in parts:
                    return float(parts.replace('s', '').strip())
        
        return elapsed
    except subprocess.TimeoutExpired:
        print(f"Timeout for grid={grid_size}, threads={num_threads}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def run_comprehensive_benchmark():
    """Run complete benchmark suite"""
    
    executable = './build/win_parallel_benchmark.exe'
    
    # Check if executable exists
    if not os.path.exists(executable.replace('./', '')):
        print(f"Executable {executable} not found!")
        print("Building...")
        os.system('g++ -std=c++11 -O2 src/win_parallel_benchmark.cpp -o build/win_parallel_benchmark.exe')
    
    # Test configurations
    grid_sizes = [64, 128, 256]
    thread_counts = [1, 2, 4, 8]
    num_steps = {64: 100, 128: 50, 256: 20}
    
    results = []
    
    print("=" * 70)
    print("COMPREHENSIVE NAVIER-STOKES BENCHMARK")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Grids: {grid_sizes}")
    print(f"Threads: {thread_counts}")
    print("=" * 70)
    
    for grid in grid_sizes:
        steps = num_steps.get(grid, 50)
        print(f"\n[Grid {grid}x{grid}, {steps} steps]")
        
        serial_time = None
        
        for threads in thread_counts:
            # Run 3 times for stability
            times = []
            for run in range(3):
                t = run_benchmark(executable, grid, threads, num_steps=steps)
                if t:
                    times.append(t)
            
            if times:
                avg_time = np.median(times)
                
                if threads == 1:
                    serial_time = avg_time
                    speedup = 1.0
                else:
                    speedup = serial_time / avg_time if serial_time else 1.0
                
                efficiency = (speedup / threads) * 100
                
                print(f"  {threads} threads: {avg_time:.3f}s | {speedup:.2f}x | {efficiency:.1f}%")
                
                results.append({
                    'grid': grid,
                    'threads': threads,
                    'steps': steps,
                    'time_s': avg_time,
                    'speedup': speedup,
                    'efficiency': efficiency
                })
    
    return results


def create_performance_plots(results):
    """Generate comprehensive performance plots"""
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    grids = df['grid'].unique()
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(grids)))
    
    # 1. Runtime vs Threads
    ax1 = axes[0, 0]
    for i, grid in enumerate(grids):
        data = df[df['grid'] == grid]
        ax1.plot(data['threads'], data['time_s'], 
                marker='o', linewidth=2, markersize=8,
                color=colors[i], label=f'{grid}×{grid}')
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Runtime vs Thread Count', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Speedup
    ax2 = axes[0, 1]
    for i, grid in enumerate(grids):
        data = df[df['grid'] == grid]
        ax2.plot(data['threads'], data['speedup'], 
                marker='s', linewidth=2, markersize=8,
                color=colors[i], label=f'{grid}×{grid}')
    
    # Ideal line
    max_threads = df['threads'].max()
    ax2.plot([1, max_threads], [1, max_threads], 
            'k--', linewidth=2, alpha=0.5, label='Ideal')
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Speedup (×)', fontsize=12)
    ax2.set_title('Parallel Speedup', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Efficiency
    ax3 = axes[1, 0]
    for i, grid in enumerate(grids):
        data = df[df['grid'] == grid]
        ax3.plot(data['threads'], data['efficiency'], 
                marker='^', linewidth=2, markersize=8,
                color=colors[i], label=f'{grid}×{grid}')
    ax3.axhline(y=100, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Number of Threads', fontsize=12)
    ax3.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax3.set_title('Parallel Efficiency', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 120])
    
    # 4. Bar chart for best speedup
    ax4 = axes[1, 1]
    best_speedups = df.groupby('grid')['speedup'].max()
    bars = ax4.bar([f'{g}×{g}' for g in best_speedups.index], 
                   best_speedups.values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Grid Size', fontsize=12)
    ax4.set_ylabel('Best Speedup (×)', fontsize=12)
    ax4.set_title('Maximum Speedup Achieved', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, best_speedups.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Navier-Stokes Parallel Performance Analysis', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = 'results/comprehensive_benchmark.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.close()
    
    return df


def save_results_csv(results, filename='results/benchmark_results.csv'):
    """Save results to CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['grid', 'threads', 'steps', 
                                               'time_s', 'speedup', 'efficiency'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved: {filename}")


def print_summary(results):
    """Print summary table"""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Grid':>10} {'Threads':>8} {'Time(s)':>10} {'Speedup':>10} {'Efficiency':>12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['grid']:>10} {r['threads']:>8} {r['time_s']:>10.3f} "
              f"{r['speedup']:>10.2f}x {r['efficiency']:>11.1f}%")
    
    print("=" * 70)
    
    # Best results
    print("\nBEST SPEEDUP PER GRID:")
    grids = set(r['grid'] for r in results)
    for grid in sorted(grids):
        grid_results = [r for r in results if r['grid'] == grid]
        best = max(grid_results, key=lambda x: x['speedup'])
        print(f"  {grid}×{grid}: {best['speedup']:.2f}× with {best['threads']} threads")


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    results = run_comprehensive_benchmark()
    
    if results:
        save_results_csv(results)
        print_summary(results)
        
        try:
            create_performance_plots(results)
        except ImportError:
            print("Note: Install pandas for visualization: pip install pandas")
    
    print("\nBenchmark complete!")
