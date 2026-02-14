"""
OpenMP Benchmark Script
AI-HPC Hybrid Project

Benchmarks the C++ solver with different thread counts and grid sizes.
Creates performance plots and scaling analysis.
"""

import subprocess
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def run_benchmark(solver_path, grid_size, threads, sim_time=0.5):
    """Run solver and extract runtime."""
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    
    cmd = [solver_path, str(sim_time), str(grid_size), '0.01']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                               env=env, timeout=600)
        
        # Parse runtime from output
        for line in result.stdout.split('\n'):
            if 'Runtime:' in line:
                match = re.search(r'Runtime:\s+([\d.]+)\s*ms', line)
                if match:
                    return float(match.group(1))
        
        print(f"Warning: Could not parse runtime. Output:\n{result.stdout[:500]}")
        return None
        
    except subprocess.TimeoutExpired:
        print(f"Timeout for grid={grid_size}, threads={threads}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def benchmark_openmp(solver_path='build/ns_main.exe', output_dir='results'):
    """Run full OpenMP benchmark suite."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Test configurations
    grid_sizes = [64, 128, 256]
    thread_counts = [1, 2, 4]
    
    # Check for more threads
    try:
        import psutil
        max_threads = psutil.cpu_count(logical=True)
        if max_threads >= 8:
            thread_counts.append(8)
    except:
        pass
    
    results = []
    
    print("=" * 60)
    print("OpenMP Benchmark Suite")
    print("=" * 60)
    
    for grid in grid_sizes:
        print(f"\nGrid size: {grid}x{grid}")
        print("-" * 40)
        
        for threads in thread_counts:
            print(f"  Testing with {threads} thread(s)...", end=" ", flush=True)
            
            time_ms = run_benchmark(solver_path, grid, threads)
            
            if time_ms is not None:
                results.append({
                    'grid': grid,
                    'threads': threads,
                    'time_ms': time_ms
                })
                print(f"{time_ms:.1f} ms")
            else:
                print("FAILED")
    
    if not results:
        print("\nNo successful benchmarks. Make sure the solver is built with OpenMP.")
        return None
    
    df = pd.DataFrame(results)
    
    # Calculate speedup relative to single thread
    for grid in grid_sizes:
        mask = df['grid'] == grid
        serial_data = df[(df['grid'] == grid) & (df['threads'] == 1)]
        
        if len(serial_data) > 0:
            serial_time = serial_data['time_ms'].values[0]
            df.loc[mask, 'speedup'] = serial_time / df.loc[mask, 'time_ms']
            df.loc[mask, 'efficiency'] = (serial_time / df.loc[mask, 'time_ms']) / df.loc[mask, 'threads'] * 100
    
    # Save results
    csv_path = os.path.join(output_dir, 'openmp_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create plots
    create_plots(df, output_dir)
    
    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    
    return df


def create_plots(df, output_dir):
    """Create benchmark visualization plots."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    grid_sizes = df['grid'].unique()
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(grid_sizes)))
    
    # Plot 1: Runtime vs Threads
    ax1 = axes[0]
    for grid, color in zip(sorted(grid_sizes), colors):
        data = df[df['grid'] == grid].sort_values('threads')
        ax1.plot(data['threads'], data['time_ms'], 'o-', 
                color=color, linewidth=2, markersize=8,
                label=f'{grid}x{grid}')
    
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Runtime (ms)', fontsize=12)
    ax1.set_title('OpenMP Performance', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['threads'].unique())
    
    # Plot 2: Speedup vs Threads
    ax2 = axes[1]
    max_threads = df['threads'].max()
    ax2.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, 
             linewidth=2, label='Ideal')
    
    for grid, color in zip(sorted(grid_sizes), colors):
        data = df[df['grid'] == grid].sort_values('threads')
        if 'speedup' in data.columns:
            ax2.plot(data['threads'], data['speedup'], 's-',
                    color=color, linewidth=2, markersize=8,
                    label=f'{grid}x{grid}')
    
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('OpenMP Scaling', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['threads'].unique())
    
    # Plot 3: Parallel Efficiency
    ax3 = axes[2]
    ax3.axhline(y=100, color='k', linestyle='--', alpha=0.5, linewidth=2, label='Ideal (100%)')
    
    for grid, color in zip(sorted(grid_sizes), colors):
        data = df[df['grid'] == grid].sort_values('threads')
        if 'efficiency' in data.columns:
            ax3.plot(data['threads'], data['efficiency'], '^-',
                    color=color, linewidth=2, markersize=8,
                    label=f'{grid}x{grid}')
    
    ax3.set_xlabel('Number of Threads', fontsize=12)
    ax3.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax3.set_title('OpenMP Efficiency', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(df['threads'].unique())
    ax3.set_ylim(0, 120)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'openmp_benchmark.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark OpenMP parallelization')
    parser.add_argument('--solver', type=str, default='build/ns_main.exe',
                       help='Path to solver executable')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Check if solver exists
    if not os.path.exists(args.solver):
        print(f"Error: Solver not found at {args.solver}")
        print("Please build the solver with OpenMP enabled:")
        print("  g++ -std=c++17 -O2 -fopenmp -I include src/core/*.cpp src/main.cpp -o build/ns_main.exe")
        return
    
    benchmark_openmp(args.solver, args.output)


if __name__ == "__main__":
    main()
