#!/usr/bin/env python3
"""
Generate Training Data for AI Model

Creates simulation data from the HPC solver for AI training.
"""

import numpy as np
import os
import sys

sys.path.insert(0, 'python')


def generate_lid_driven_cavity(nx=64, ny=64, num_steps=500, dt=0.001, Re=100.0):
    """
    Generate lid-driven cavity simulation data using Python solver.
    """
    print(f"Generating {nx}×{ny} simulation with Re={Re}...")
    
    # Initialize
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    nu = 1.0 / Re
    
    # Storage for snapshots
    snapshots_u = []
    snapshots_v = []
    snapshots_p = []
    
    # Time stepping
    for step in range(num_steps):
        # Simple diffusion step (for data generation)
        u_new = u.copy()
        v_new = v.copy()
        
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                d2udx2 = (u[j, i+1] - 2*u[j, i] + u[j, i-1]) / dx**2
                d2udy2 = (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / dy**2
                d2vdx2 = (v[j, i+1] - 2*v[j, i] + v[j, i-1]) / dx**2
                d2vdy2 = (v[j+1, i] - 2*v[j, i] + v[j-1, i]) / dy**2
                
                u_new[j, i] = u[j, i] + dt * nu * (d2udx2 + d2udy2)
                v_new[j, i] = v[j, i] + dt * nu * (d2vdx2 + d2vdy2)
        
        u = u_new
        v = v_new
        
        # Boundary conditions
        u[0, :] = 0; u[-1, :] = 1.0  # Lid velocity
        u[:, 0] = 0; u[:, -1] = 0
        v[0, :] = 0; v[-1, :] = 0
        v[:, 0] = 0; v[:, -1] = 0
        
        # Save every 10 steps
        if step % 10 == 0:
            snapshots_u.append(u.copy())
            snapshots_v.append(v.copy())
            snapshots_p.append(p.copy())
            
            if step % 100 == 0:
                print(f"  Step {step}/{num_steps}")
    
    return np.array(snapshots_u), np.array(snapshots_v), np.array(snapshots_p)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate training data')
    parser.add_argument('--grid_size', type=int, default=64, help='Grid size')
    parser.add_argument('--num_timesteps', type=int, default=500, help='Number of timesteps')
    parser.add_argument('--reynolds', type=float, default=100.0, help='Reynolds number')
    args = parser.parse_args()
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    u, v, p = generate_lid_driven_cavity(
        nx=args.grid_size, 
        ny=args.grid_size,
        num_steps=args.num_timesteps,
        Re=args.reynolds
    )
    
    # Save
    filename = f'data/simulation_{args.grid_size}x{args.grid_size}.npz'
    np.savez(filename, u=u, v=v, p=p)
    print(f"\n✓ Saved: {filename}")
    print(f"  U shape: {u.shape}")
    print(f"  V shape: {v.shape}")
    print(f"  P shape: {p.shape}")


if __name__ == "__main__":
    main()
