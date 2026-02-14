#!/usr/bin/env python3
"""
Large Scale Demo - Shows solver scaling to large grids
"""

import subprocess
import os
import time
import sys

def run_large_scale_demo():
    """Demonstrate scaling to production sizes"""
    
    print("=" * 70)
    print("NAVIER-STOKES LARGE-SCALE DEMONSTRATION")
    print("=" * 70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M')}")
    print()
    
    executable = './build/win_parallel_benchmark.exe'
    
    # Check executable
    if not os.path.exists(executable.replace('./', '')):
        print("Building executable...")
        os.system('g++ -std=c++11 -O2 src/win_parallel_benchmark.cpp -o build/win_parallel_benchmark.exe')
    
    # Configurations: (grid_size, num_steps, threads, description)
    configs = [
        (64,  100, 2, "Small (64×64)"),
        (128, 50,  4, "Medium (128×128)"),
        (256, 20,  8, "Large (256×256)"),
    ]
    
    results = []
    
    for grid, steps, threads, desc in configs:
        print(f"\n{desc}")
        print(f"  Grid: {grid}×{grid} = {grid*grid:,} points")
        print(f"  Steps: {steps}")
        print(f"  Threads: {threads}")
        print("-" * 50)
        
        cmd = [executable, str(grid), str(steps), str(threads)]
        
        start = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            elapsed = time.time() - start
            
            print(f"  Wall-clock time: {elapsed:.2f}s")
            print(f"  Memory estimate: ~{grid*grid*3*8/1e6:.1f} MB (for u,v,p)")
            print(f"  Throughput: {grid*grid*steps/elapsed/1e6:.2f} M points/sec")
            
            results.append({
                'grid': grid,
                'steps': steps,
                'threads': threads,
                'time': elapsed,
                'throughput': grid*grid*steps/elapsed
            })
            
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (>300s)")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Grid':>12} {'Steps':>8} {'Threads':>8} {'Time(s)':>10} {'Throughput':>15}")
    print("-" * 70)
    for r in results:
        print(f"{r['grid']:>12} {r['steps']:>8} {r['threads']:>8} "
              f"{r['time']:>10.2f} {r['throughput']/1e6:>14.2f} M/s")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_large_scale_demo()
