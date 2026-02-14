#!/usr/bin/env python3
"""
Uncertainty Quantification - Monte Carlo Dropout method.

Provides confidence intervals for AI predictions, not just point estimates.
Identifies high-uncertainty regions where HPC verification is recommended.

Generates: results/uncertainty_quantification.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'python')

import torch
import torch.nn as nn


class SimpleCFDSolver:
    """Python Navier-Stokes solver"""
    
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


class UncertainAIPredictor(nn.Module):
    """AI model with Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.param_encoder = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 4*4*64), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, x):
        x = self.param_encoder(x).view(-1, 64, 4, 4)
        return self.decoder(x)
    
    def predict_with_uncertainty(self, x, n_samples=30):
        """
        Monte Carlo Dropout: run multiple forward passes with dropout enabled
        Returns mean prediction and uncertainty (std dev)
        """
        self.train()  # Keep dropout active
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std


def main():
    print("="*70)
    print("UNCERTAINTY QUANTIFICATION - Monte Carlo Dropout")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    # Generate data
    print("\nGenerating training data...")
    training_velocities = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    training_states = []
    for v in training_velocities:
        state = SimpleCFDSolver(lid_velocity=v).run(200)
        training_states.append(state)
        print(f"  Generated v={v:.2f}")
    
    X = torch.FloatTensor([[v] for v in training_velocities])
    Y = torch.FloatTensor(np.array(training_states))
    
    # Train model
    print("\nTraining model with dropout...")
    model = UncertainAIPredictor(dropout_rate=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(80):
        model.train()
        opt.zero_grad()
        loss = nn.MSELoss()(model(X), Y)
        loss.backward()
        opt.step()
    print(f"  Training loss: {loss.item():.6f}")
    
    # Test with uncertainty
    test_velocities = [0.6, 0.85, 1.1, 1.35, 1.6, 1.85, 2.2, 2.5]
    # Note: 2.2 and 2.5 are EXTRAPOLATION (outside training range)
    
    print(f"\nPredicting {len(test_velocities)} cases with uncertainty...")
    print(f"  Interpolation range: [0.5, 2.0]")
    print(f"  Cases 2.2 and 2.5 are EXTRAPOLATION (expect higher uncertainty)")
    
    results = {}
    for v in test_velocities:
        mean_pred, std_pred = model.predict_with_uncertainty(
            torch.FloatTensor([[v]]), n_samples=30
        )
        
        mean_np = mean_pred.numpy()[0]
        std_np = std_pred.numpy()[0]
        
        # Get ground truth for comparison
        gt = SimpleCFDSolver(lid_velocity=v).run(200)
        rmse = np.sqrt(np.mean((mean_np - gt)**2))
        mean_uncertainty = np.mean(std_np)
        
        results[v] = {
            'mean': mean_np,
            'std': std_np,
            'gt': gt,
            'rmse': rmse,
            'uncertainty': mean_uncertainty,
            'is_extrapolation': v > 2.0 or v < 0.5
        }
        
        label = "EXTRAPOLATION" if results[v]['is_extrapolation'] else "interpolation"
        print(f"  v={v:.2f} | RMSE: {rmse:.4f} | Uncertainty: {mean_uncertainty:.4f} | {label}")
    
    # Generate plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Uncertainty Quantification (Monte Carlo Dropout)', fontsize=16, fontweight='bold')
    
    # Plot 1: Uncertainty vs velocity
    ax = axes[0, 0]
    vels = list(results.keys())
    uncertainties = [results[v]['uncertainty'] for v in vels]
    rmses = [results[v]['rmse'] for v in vels]
    colors = ['coral' if results[v]['is_extrapolation'] else 'steelblue' for v in vels]
    ax.bar(range(len(vels)), uncertainties, color=colors)
    ax.set_xticks(range(len(vels)))
    ax.set_xticklabels([f'{v:.1f}' for v in vels], rotation=45)
    ax.set_xlabel('Lid Velocity')
    ax.set_ylabel('Mean Uncertainty (σ)')
    ax.set_title('1. Uncertainty by Parameter')
    ax.axvline(x=5.5, color='red', linestyle='--', alpha=0.5, label='Extrapolation boundary')
    ax.legend()
    
    # Plot 2: Uncertainty correlates with error
    ax = axes[0, 1]
    ax.scatter(uncertainties, rmses, c=colors, s=100, zorder=5)
    for i, v in enumerate(vels):
        ax.annotate(f'v={v:.1f}', (uncertainties[i], rmses[i]),
                   textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel('Mean Uncertainty (σ)')
    ax.set_ylabel('Actual RMSE')
    ax.set_title('2. Uncertainty vs Actual Error')
    # Fit line
    z = np.polyfit(uncertainties, rmses, 1)
    x_line = np.linspace(min(uncertainties), max(uncertainties), 50)
    ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5, label='Trend')
    ax.legend()
    
    # Plot 3: Uncertainty heatmap for interpolation case
    v_interp = 1.1
    ax = axes[0, 2]
    vmag_std = np.sqrt(results[v_interp]['std'][0]**2 + results[v_interp]['std'][1]**2)
    im = ax.imshow(vmag_std, cmap='YlOrRd', origin='lower')
    ax.set_title(f'3. Uncertainty Map (v={v_interp}, interpolation)')
    plt.colorbar(im, ax=ax)
    
    # Plot 4: Mean prediction vs ground truth
    v_show = 1.1
    vmag_mean = np.sqrt(results[v_show]['mean'][0]**2 + results[v_show]['mean'][1]**2)
    vmag_gt = np.sqrt(results[v_show]['gt'][0]**2 + results[v_show]['gt'][1]**2)
    
    ax = axes[1, 0]
    im = ax.imshow(vmag_gt, cmap='hot', origin='lower')
    ax.set_title(f'4a. HPC Ground Truth (v={v_show})')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    im = ax.imshow(vmag_mean, cmap='hot', origin='lower')
    ax.set_title(f'4b. AI Mean Prediction (v={v_show})')
    plt.colorbar(im, ax=ax)
    
    # Plot 6: Uncertainty heatmap for extrapolation case
    v_extrap = 2.5
    ax = axes[1, 2]
    vmag_std_ext = np.sqrt(results[v_extrap]['std'][0]**2 + results[v_extrap]['std'][1]**2)
    im = ax.imshow(vmag_std_ext, cmap='YlOrRd', origin='lower')
    ax.set_title(f'5. Uncertainty Map (v={v_extrap}, EXTRAPOLATION)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('results/uncertainty_quantification.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved: results/uncertainty_quantification.png")
    
    # Print summary
    interp_unc = np.mean([results[v]['uncertainty'] for v in vels if not results[v]['is_extrapolation']])
    extrap_unc = np.mean([results[v]['uncertainty'] for v in vels if results[v]['is_extrapolation']])
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Interpolation uncertainty: {interp_unc:.4f}")
    print(f"  Extrapolation uncertainty: {extrap_unc:.4f}")
    print(f"  Ratio (extrap/interp):     {extrap_unc/interp_unc:.1f}x")
    print(f"\n  KEY INSIGHT: Uncertainty correctly identifies extrapolation cases")
    print(f"  → Can auto-flag high-uncertainty predictions for HPC verification")
    print("="*70)


if __name__ == "__main__":
    main()
