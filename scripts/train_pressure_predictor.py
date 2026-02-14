#!/usr/bin/env python3
"""
Train AI Pressure Predictor on ACTUAL Pressure Poisson Solutions

The key insight: We need to train on (velocity divergence → pressure solution) pairs,
not just (u,v → p) from simulation data.

This script:
1. Generates velocity fields
2. Computes actual pressure solutions via iterative solver
3. Trains AI to predict pressure from velocity
4. Validates iteration reduction
"""

import numpy as np
import time
import os
import sys

sys.path.insert(0, 'python')

import torch
import torch.nn as nn
import torch.optim as optim


class TinyPressureNet(nn.Module):
    """Tiny CNN for pressure prediction"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


def solve_pressure_exact(u, v, dx, dy, dt=0.001, tol=1e-6, max_iter=1000):
    """
    Solve pressure Poisson equation exactly.
    Returns the converged pressure field.
    """
    ny, nx = u.shape
    
    # Compute RHS: div(u*)/dt
    rhs = np.zeros((ny, nx))
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            dudx = (u[j, i+1] - u[j, i-1]) / (2*dx)
            dvdy = (v[j+1, i] - v[j-1, i]) / (2*dy)
            rhs[j, i] = (dudx + dvdy) / dt
    
    # Solve via Red-Black Gauss-Seidel
    p = np.zeros((ny, nx))
    dx2, dy2 = dx**2, dy**2
    coeff = 1.0 / (2/dx2 + 2/dy2)
    
    for it in range(max_iter):
        max_change = 0.0
        
        for color in [0, 1]:
            for j in range(1, ny-1):
                for i in range(1, nx-1):
                    if (i + j) % 2 == color:
                        p_new = coeff * (
                            (p[j, i+1] + p[j, i-1]) / dx2 +
                            (p[j+1, i] + p[j-1, i]) / dy2 -
                            rhs[j, i]
                        )
                        max_change = max(max_change, abs(p_new - p[j, i]))
                        p[j, i] = p_new
        
        if max_change < tol:
            break
    
    return p


def generate_training_data(num_samples=500, nx=64, ny=64):
    """
    Generate (velocity, pressure) pairs where pressure is the EXACT solution.
    """
    print(f"Generating {num_samples} training samples...")
    
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    
    X = []  # Velocities
    Y = []  # Pressures
    
    for i in range(num_samples):
        if (i + 1) % 50 == 0:
            print(f"  Sample {i+1}/{num_samples}")
        
        # Random velocity field (various patterns)
        pattern = np.random.randint(0, 4)
        
        u = np.zeros((ny, nx))
        v = np.zeros((ny, nx))
        
        for j in range(ny):
            for ii in range(nx):
                x, y = ii * dx, j * dy
                
                if pattern == 0:
                    # Lid-driven cavity-like
                    u[j, ii] = np.sin(np.pi * y) * np.random.uniform(0.5, 1.5)
                    v[j, ii] = 0.0
                elif pattern == 1:
                    # Vortex
                    cx, cy = 0.5, 0.5
                    r = np.sqrt((x-cx)**2 + (y-cy)**2) + 0.01
                    u[j, ii] = -(y - cy) / r * np.exp(-r*5)
                    v[j, ii] = (x - cx) / r * np.exp(-r*5)
                elif pattern == 2:
                    # Shear
                    u[j, ii] = np.sin(2 * np.pi * y)
                    v[j, ii] = np.sin(2 * np.pi * x) * 0.5
                else:
                    # Random smooth
                    freq = np.random.randint(1, 4)
                    u[j, ii] = np.sin(freq * np.pi * x) * np.cos(freq * np.pi * y)
                    v[j, ii] = -np.cos(freq * np.pi * x) * np.sin(freq * np.pi * y)
        
        # Apply boundary conditions
        u[0, :] = 0; u[-1, :] = 0; u[:, 0] = 0; u[:, -1] = 0
        v[0, :] = 0; v[-1, :] = 0; v[:, 0] = 0; v[:, -1] = 0
        
        # Add some noise for robustness
        u += np.random.randn(ny, nx) * 0.01
        v += np.random.randn(ny, nx) * 0.01
        
        # Solve for EXACT pressure
        p = solve_pressure_exact(u, v, dx, dy)
        
        X.append(np.stack([u, v]))
        Y.append(p[np.newaxis, :, :])
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    
    return X, Y


def train_model(X, Y, epochs=50, batch_size=32):
    """Train the pressure predictor"""
    
    print(f"\nTraining on {X.shape[0]} samples for {epochs} epochs...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = TinyPressureNet().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)
    
    num_batches = X.shape[0] // batch_size
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Shuffle
        perm = torch.randperm(X_tensor.shape[0])
        X_shuffled = X_tensor[perm]
        Y_shuffled = Y_tensor[perm]
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            
            batch_x = X_shuffled[start:end]
            batch_y = Y_shuffled[start:end]
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'num_params': num_params
            }, 'checkpoints/pressure_predictor.pth')
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    
    print(f"\n✓ Best loss: {best_loss:.6f}")
    print(f"✓ Saved to checkpoints/pressure_predictor.pth")
    
    return model


def validate_model():
    """Validate iteration reduction"""
    
    print("\n" + "="*60)
    print("VALIDATION: Measuring Iteration Reduction")
    print("="*60)
    
    # Load model
    model = TinyPressureNet()
    checkpoint = torch.load('checkpoints/pressure_predictor.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test parameters
    nx, ny = 64, 64
    dx = dy = 1.0 / (nx - 1)
    
    # Generate test velocity
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            x, y = i * dx, j * dy
            u[j, i] = np.sin(np.pi * y)
            v[j, i] = -np.sin(np.pi * x) * np.cos(np.pi * y) * 0.5
    
    # Get target pressure (exact solution)
    p_exact = solve_pressure_exact(u, v, dx, dy, tol=1e-8)
    
    # AI prediction
    velocity = torch.FloatTensor(np.stack([u, v])[np.newaxis, :, :, :])
    with torch.no_grad():
        p_ai = model(velocity).squeeze().numpy()
    
    # Count iterations for zero vs AI initial guess
    def count_iterations(p_init, tol=1e-5):
        p = p_init.copy()
        dx2, dy2 = dx**2, dy**2
        coeff = 1.0 / (2/dx2 + 2/dy2)
        
        rhs = np.zeros((ny, nx))
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                dudx = (u[j, i+1] - u[j, i-1]) / (2*dx)
                dvdy = (v[j+1, i] - v[j-1, i]) / (2*dy)
                rhs[j, i] = (dudx + dvdy) / 0.001
        
        for it in range(500):
            max_change = 0.0
            for color in [0, 1]:
                for j in range(1, ny-1):
                    for i in range(1, nx-1):
                        if (i + j) % 2 == color:
                            p_new = coeff * (
                                (p[j, i+1] + p[j, i-1]) / dx2 +
                                (p[j+1, i] + p[j-1, i]) / dy2 -
                                rhs[j, i]
                            )
                            max_change = max(max_change, abs(p_new - p[j, i]))
                            p[j, i] = p_new
            if max_change < tol:
                return it + 1
        return 500
    
    iters_zero = count_iterations(np.zeros((ny, nx)))
    iters_ai = count_iterations(p_ai)
    
    reduction = (1 - iters_ai / iters_zero) * 100
    
    print(f"\nIterations from zero:     {iters_zero}")
    print(f"Iterations from AI:       {iters_ai}")
    print(f"Reduction:                {reduction:.1f}%")
    
    if reduction > 50:
        print(f"\n✅ AI reduces iterations by {reduction:.0f}% - SUCCESS!")
    elif reduction > 0:
        print(f"\n⚠️  AI reduces iterations by {reduction:.0f}% - some benefit")
    else:
        print(f"\n❌ AI not helping - need more training")
    
    return reduction


if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    
    print("="*60)
    print("PRESSURE PREDICTOR TRAINING")
    print("Training on EXACT pressure solutions")
    print("="*60)
    
    # Generate training data
    X, Y = generate_training_data(num_samples=300, nx=64, ny=64)
    
    # Train
    model = train_model(X, Y, epochs=50)
    
    # Validate
    reduction = validate_model()
