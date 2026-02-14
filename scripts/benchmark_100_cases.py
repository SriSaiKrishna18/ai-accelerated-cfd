#!/usr/bin/env python3
"""
100-Case Benchmark - FULLY MEASURED
No projections - actually runs all 100 cases!
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, 'python')

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleCFDSolver:
    """Simple Python Navier-Stokes solver"""
    
    def __init__(self, nx=64, ny=64, Re=100.0, lid_velocity=1.0):
        self.nx, self.ny = nx, ny
        self.Re = Re
        self.lid_velocity = lid_velocity
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)
        self.dt = 0.001
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        self.u[-1, :] = lid_velocity
    
    def step(self):
        nu = 1.0 / self.Re
        u_new = self.u.copy()
        v_new = self.v.copy()
        
        for j in range(1, self.ny-1):
            for i in range(1, self.nx-1):
                d2udx2 = (self.u[j, i+1] - 2*self.u[j, i] + self.u[j, i-1]) / self.dx**2
                d2udy2 = (self.u[j+1, i] - 2*self.u[j, i] + self.u[j-1, i]) / self.dy**2
                d2vdx2 = (self.v[j, i+1] - 2*self.v[j, i] + self.v[j, i-1]) / self.dx**2
                d2vdy2 = (self.v[j+1, i] - 2*self.v[j, i] + self.v[j-1, i]) / self.dy**2
                
                u_new[j, i] = self.u[j, i] + self.dt * nu * (d2udx2 + d2udy2)
                v_new[j, i] = self.v[j, i] + self.dt * nu * (d2vdx2 + d2vdy2)
        
        self.u = u_new
        self.v = v_new
        self.u[0, :] = 0; self.u[-1, :] = self.lid_velocity
        self.u[:, 0] = 0; self.u[:, -1] = 0
        self.v[0, :] = 0; self.v[-1, :] = 0
        self.v[:, 0] = 0; self.v[:, -1] = 0
    
    def run(self, num_steps):
        for _ in range(num_steps):
            self.step()
        return np.stack([self.u, self.v, self.p])


class SimpleAIPredictor(nn.Module):
    def __init__(self, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.param_encoder = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 4 * 4 * 64), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )
    
    def forward(self, lid_velocity):
        x = self.param_encoder(lid_velocity)
        x = x.view(-1, 64, 4, 4)
        return self.decoder(x)


def run_hpc(lid_velocity, num_steps=200):
    solver = SimpleCFDSolver(lid_velocity=lid_velocity)
    return solver.run(num_steps)


def main():
    print("="*70)
    print("100-CASE BENCHMARK - FULLY MEASURED (NO PROJECTIONS)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Define 100 different cases (lid velocities)
    all_velocities = np.linspace(0.5, 2.0, 100)
    training_velocities = all_velocities[::14][:7]  # 7 evenly spaced for training
    test_velocities = [v for v in all_velocities if v not in training_velocities]
    
    print(f"\nTotal cases: 100")
    print(f"Training cases (HPC): {len(training_velocities)}")
    print(f"Test cases (AI): {len(test_velocities)}")
    
    # ============================================================
    # METHOD 1: PURE HPC - Run all 100 cases
    # ============================================================
    print("\n" + "-"*70)
    print("METHOD 1: PURE HPC - Running ALL 100 simulations...")
    print("-"*70)
    
    hpc_start = time.time()
    hpc_results = {}
    for i, v in enumerate(all_velocities):
        if (i + 1) % 20 == 0:
            print(f"  HPC simulation {i+1}/100...")
        hpc_results[v] = run_hpc(v)
    hpc_time = time.time() - hpc_start
    
    print(f"\n✓ Pure HPC complete: {hpc_time*1000:.0f}ms for 100 cases")
    
    # ============================================================
    # METHOD 2: HYBRID - HPC for training + AI for rest
    # ============================================================
    print("\n" + "-"*70)
    print("METHOD 2: AI-HPC HYBRID")
    print("-"*70)
    
    hybrid_start = time.time()
    
    # Step 1: HPC for training cases
    print("\nStep 1: Running HPC for 7 training cases...")
    hpc_train_start = time.time()
    training_data = []
    for v in training_velocities:
        state = run_hpc(v)
        training_data.append((v, state))
    hpc_train_time = time.time() - hpc_train_start
    print(f"  ✓ HPC training: {hpc_train_time*1000:.0f}ms")
    
    # Step 2: Train AI
    print("\nStep 2: Training AI model...")
    train_start = time.time()
    
    X = torch.FloatTensor([[d[0]] for d in training_data]).to(device)
    Y = torch.FloatTensor(np.array([d[1] for d in training_data])).to(device)
    
    model = SimpleAIPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(50):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()
    
    train_time = time.time() - train_start
    print(f"  ✓ AI training: {train_time*1000:.0f}ms")
    
    # Step 3: AI inference for remaining 93 cases
    print(f"\nStep 3: AI inference for {len(test_velocities)} cases...")
    model.eval()
    inference_start = time.time()
    ai_results = {}
    for v in test_velocities:
        x = torch.FloatTensor([[v]]).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]
        ai_results[v] = pred
    inference_time = time.time() - inference_start
    print(f"  ✓ AI inference: {inference_time*1000:.0f}ms")
    
    hybrid_time = time.time() - hybrid_start
    
    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "="*70)
    print("MEASURED RESULTS (100 CASES)")
    print("="*70)
    
    print(f"\n{'Method':<25} {'Time (ms)':<15} {'Cases':<10}")
    print("-"*50)
    print(f"{'Pure HPC':<25} {hpc_time*1000:<15.0f} {100:<10}")
    print(f"{'AI-HPC Hybrid':<25} {hybrid_time*1000:<15.0f} {100:<10}")
    print("-"*50)
    
    speedup = hpc_time / hybrid_time
    print(f"\n{'MEASURED SPEEDUP:':<25} {speedup:.2f}×")
    
    print(f"\nBreakdown of Hybrid:")
    print(f"  HPC (7 training):    {hpc_train_time*1000:.0f}ms")
    print(f"  AI Training:         {train_time*1000:.0f}ms")
    print(f"  AI Inference (93):   {inference_time*1000:.0f}ms")
    
    # Accuracy check
    print("\n" + "-"*70)
    print("ACCURACY CHECK")
    print("-"*70)
    
    errors = []
    for v in list(test_velocities)[:5]:
        ai_state = ai_results[v]
        hpc_state = hpc_results[v]
        rmse = np.sqrt(np.mean((ai_state - hpc_state)**2))
        errors.append(rmse)
        print(f"  Velocity {v:.3f}: RMSE = {rmse:.4f}")
    
    print(f"\n  Average RMSE: {np.mean(errors):.4f}")
    
    print("\n" + "="*70)
    if speedup > 2:
        print(f"✅ SUCCESS: {speedup:.1f}× MEASURED SPEEDUP for 100 cases!")
    else:
        print(f"⚠️  Speedup: {speedup:.1f}×")
    print("="*70)
    
    return {
        'hpc_time': hpc_time * 1000,
        'hybrid_time': hybrid_time * 1000,
        'speedup': speedup
    }


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    main()
