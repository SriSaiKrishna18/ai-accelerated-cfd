"""
Data Generation Script
AI-HPC Hybrid Project

Runs the C++ HPC solver to generate training data for the AI model.
Saves sequences of simulation states to HDF5/numpy format.
"""

import os
import sys
import subprocess
import struct
import argparse
import numpy as np
from pathlib import Path


def run_solver(nx=128, ny=128, nu=0.01, t_final=1.0, checkpoint_interval=0.01,
               output_dir="data/training", solver_path="build/ns_main"):
    """
    Run the C++ solver multiple times to generate training data.
    
    Args:
        nx, ny: Grid resolution
        nu: Viscosity (determines Reynolds number)
        t_final: Total simulation time
        checkpoint_interval: Time between checkpoints
        output_dir: Where to save data
        solver_path: Path to compiled solver executable
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating data with:")
    print(f"  Grid: {nx}x{ny}")
    print(f"  Viscosity: {nu}")
    print(f"  Reynolds: {1.0/nu:.0f}")
    print(f"  Final time: {t_final}")
    print(f"  Interval: {checkpoint_interval}")
    
    # Check if solver exists
    if not os.path.exists(solver_path):
        print(f"Error: Solver not found at {solver_path}")
        print("Please build the C++ solver first:")
        print("  mkdir build && cd build && cmake .. && make")
        return None
    
    # Generate data at multiple time points
    all_u = []
    all_v = []
    all_p = []
    times = []
    
    num_steps = int(t_final / checkpoint_interval)
    
    for i in range(1, num_steps + 1):
        t = i * checkpoint_interval
        
        print(f"Running solver to t = {t:.3f}...")
        
        # Run solver
        result = subprocess.run(
            [solver_path, str(t), str(nx), str(nu)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Solver failed: {result.stderr}")
            continue
        
        # Load checkpoint
        checkpoint_path = "data/checkpoints/checkpoint.bin"
        if os.path.exists(checkpoint_path):
            u, v, p, time, step = load_checkpoint(checkpoint_path, nx, ny)
            all_u.append(u)
            all_v.append(v)
            all_p.append(p)
            times.append(time)
            print(f"  Loaded checkpoint: t={time:.4f}, step={step}")
    
    if len(all_u) == 0:
        print("No data generated!")
        return None
    
    # Stack arrays
    u_array = np.array(all_u, dtype=np.float32)
    v_array = np.array(all_v, dtype=np.float32)
    p_array = np.array(all_p, dtype=np.float32)
    times_array = np.array(times, dtype=np.float32)
    
    # Save to numpy
    output_path = os.path.join(output_dir, "simulation_data.npz")
    np.savez(output_path, u=u_array, v=v_array, p=p_array, times=times_array)
    
    print(f"\nData saved to {output_path}")
    print(f"Shape: ({len(all_u)}, {ny}, {nx})")
    
    return output_path


def load_checkpoint(filepath, nx, ny):
    """Load state from binary checkpoint file."""
    with open(filepath, 'rb') as f:
        # Read metadata
        nx_file = struct.unpack('i', f.read(4))[0]
        ny_file = struct.unpack('i', f.read(4))[0]
        time = struct.unpack('d', f.read(8))[0]
        step = struct.unpack('i', f.read(4))[0]
        
        # Read state arrays
        n = nx_file * ny_file
        u = np.array(struct.unpack(f'{n}d', f.read(n * 8)))
        v = np.array(struct.unpack(f'{n}d', f.read(n * 8)))
        p = np.array(struct.unpack(f'{n}d', f.read(n * 8)))
        
        # Reshape to 2D (column-major order from C++)
        u = u.reshape((ny_file, nx_file))
        v = v.reshape((ny_file, nx_file))
        p = p.reshape((ny_file, nx_file))
        
    return u, v, p, time, step


def generate_synthetic_data(output_dir="data/training", num_trajectories=10,
                           frames_per_trajectory=100, nx=64, ny=64):
    """
    Generate synthetic Taylor-Green vortex data without running the solver.
    Useful for testing the AI model pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating synthetic data:")
    print(f"  Trajectories: {num_trajectories}")
    print(f"  Frames per trajectory: {frames_per_trajectory}")
    print(f"  Grid: {nx}x{ny}")
    
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    X, Y = np.meshgrid(x, y)
    
    for traj_idx in range(num_trajectories):
        # Random viscosity for variety
        nu = np.random.uniform(0.005, 0.05)
        
        # Random amplitude and phase
        amp = np.random.uniform(0.8, 1.2)
        phase_x = np.random.uniform(0, 0.5)
        phase_y = np.random.uniform(0, 0.5)
        
        u = np.zeros((frames_per_trajectory, ny, nx), dtype=np.float32)
        v = np.zeros((frames_per_trajectory, ny, nx), dtype=np.float32)
        p = np.zeros((frames_per_trajectory, ny, nx), dtype=np.float32)
        
        for t in range(frames_per_trajectory):
            time = t * 0.01
            decay = np.exp(-2 * nu * time)
            
            u[t] = amp * np.sin(X + phase_x) * np.cos(Y + phase_y) * decay
            v[t] = -amp * np.cos(X + phase_x) * np.sin(Y + phase_y) * decay
            p[t] = -0.25 * amp**2 * (np.cos(2*(X + phase_x)) + np.cos(2*(Y + phase_y))) * decay**2
        
        output_path = os.path.join(output_dir, f"trajectory_{traj_idx:03d}.npz")
        np.savez(output_path, u=u, v=v, p=p, nu=nu)
        print(f"  Saved: {output_path}")
    
    # Also create combined train/val split
    all_u = []
    all_v = []
    all_p = []
    
    for traj_idx in range(num_trajectories):
        data = np.load(os.path.join(output_dir, f"trajectory_{traj_idx:03d}.npz"))
        all_u.append(data['u'])
        all_v.append(data['v'])
        all_p.append(data['p'])
    
    # Concatenate all trajectories
    all_u = np.concatenate(all_u, axis=0)
    all_v = np.concatenate(all_v, axis=0)
    all_p = np.concatenate(all_p, axis=0)
    
    # Split 80/20
    n_total = len(all_u)
    n_train = int(0.8 * n_total)
    
    np.savez(os.path.join(output_dir, "train_data.npz"),
             u=all_u[:n_train], v=all_v[:n_train], p=all_p[:n_train])
    np.savez(os.path.join(output_dir, "val_data.npz"),
             u=all_u[n_train:], v=all_v[n_train:], p=all_p[n_train:])
    
    print(f"\nCreated train_data.npz ({n_train} frames)")
    print(f"Created val_data.npz ({n_total - n_train} frames)")


def main():
    parser = argparse.ArgumentParser(description="Generate training data for AI model")
    
    parser.add_argument('--mode', type=str, default='synthetic',
                       choices=['hpc', 'synthetic'],
                       help='Generation mode: hpc (run solver) or synthetic')
    parser.add_argument('--output-dir', type=str, default='data/training',
                       help='Output directory')
    parser.add_argument('--nx', type=int, default=64, help='Grid size x')
    parser.add_argument('--ny', type=int, default=64, help='Grid size y')
    parser.add_argument('--nu', type=float, default=0.01, help='Viscosity')
    parser.add_argument('--t-final', type=float, default=1.0, help='Final time')
    parser.add_argument('--num-trajectories', type=int, default=10,
                       help='Number of trajectories for synthetic data')
    parser.add_argument('--frames', type=int, default=100,
                       help='Frames per trajectory')
    
    args = parser.parse_args()
    
    if args.mode == 'hpc':
        run_solver(
            nx=args.nx, ny=args.ny, nu=args.nu,
            t_final=args.t_final, output_dir=args.output_dir
        )
    else:
        generate_synthetic_data(
            output_dir=args.output_dir,
            num_trajectories=args.num_trajectories,
            frames_per_trajectory=args.frames,
            nx=args.nx, ny=args.ny
        )


if __name__ == "__main__":
    main()
