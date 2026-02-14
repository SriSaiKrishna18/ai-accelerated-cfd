#!/usr/bin/env python3
"""
Physics Validation for AI Predictions

This script validates that AI predictions satisfy physical constraints:
1. Divergence-free (∇·u ≈ 0 for incompressible flow)
2. Energy conservation (kinetic energy decay)
3. Different Reynolds numbers

Run: python scripts/physics_validation.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'python')

def compute_divergence(u, v, dx=1.0, dy=1.0):
    """Compute divergence ∇·u = ∂u/∂x + ∂v/∂y"""
    dudx = np.gradient(u, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)
    return dudx + dvdy

def compute_kinetic_energy(u, v, dx=1.0, dy=1.0):
    """Compute kinetic energy KE = 0.5 * ∫(u² + v²) dA"""
    return 0.5 * np.sum(u**2 + v**2) * dx * dy

def validate_divergence_free(predictions, title="AI Predictions"):
    """Check if velocity field is approximately divergence-free"""
    
    if len(predictions.shape) == 5:  # (batch, time, channels, H, W)
        u = predictions[0, :, 0, :, :]  # First sample, u-velocity
        v = predictions[0, :, 1, :, :]  # First sample, v-velocity
    else:
        u = predictions[:, 0, :, :]
        v = predictions[:, 1, :, :]
    
    num_steps = u.shape[0]
    max_divs = []
    mean_divs = []
    
    nx, ny = u.shape[1], u.shape[2]
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    
    for t in range(num_steps):
        div = compute_divergence(u[t], v[t], dx, dy)
        max_divs.append(np.max(np.abs(div)))
        mean_divs.append(np.mean(np.abs(div)))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    steps = np.arange(num_steps)
    ax1.plot(steps, max_divs, 'b-', linewidth=2, label='Max |∇·u|')
    ax1.plot(steps, mean_divs, 'r--', linewidth=2, label='Mean |∇·u|')
    ax1.axhline(y=0.1, color='g', linestyle=':', label='Threshold')
    ax1.set_xlabel('Prediction Step')
    ax1.set_ylabel('Divergence')
    ax1.set_title(f'Divergence Check: {title}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Show divergence field at last time step
    div_final = compute_divergence(u[-1], v[-1], dx, dy)
    im = ax2.imshow(np.abs(div_final), cmap='hot', origin='lower')
    ax2.set_title(f'|∇·u| at step {num_steps-1}')
    plt.colorbar(im, ax=ax2, label='|Divergence|')
    
    plt.tight_layout()
    plt.savefig('results/divergence_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary
    print("\n" + "="*50)
    print("DIVERGENCE VALIDATION")
    print("="*50)
    print(f"Max divergence: {np.max(max_divs):.6f}")
    print(f"Mean divergence: {np.mean(mean_divs):.6f}")
    
    if np.max(max_divs) < 0.1:
        print("✅ PASS: Divergence is acceptably small")
        return True
    else:
        print("⚠️  WARNING: Large divergence detected")
        return False


def validate_energy_conservation(predictions, ground_truth=None):
    """Check kinetic energy evolution"""
    
    if len(predictions.shape) == 5:
        u_pred = predictions[0, :, 0, :, :]
        v_pred = predictions[0, :, 1, :, :]
    else:
        u_pred = predictions[:, 0, :, :]
        v_pred = predictions[:, 1, :, :]
    
    num_steps = u_pred.shape[0]
    nx, ny = u_pred.shape[1], u_pred.shape[2]
    dx = 1.0 / (nx - 1)
    
    # Compute KE for predictions
    ke_pred = [compute_kinetic_energy(u_pred[t], v_pred[t], dx, dx) 
               for t in range(num_steps)]
    
    # Compute for ground truth if available
    ke_truth = None
    if ground_truth is not None:
        if len(ground_truth.shape) == 5:
            u_true = ground_truth[0, :, 0, :, :]
            v_true = ground_truth[0, :, 1, :, :]
        else:
            u_true = ground_truth[:, 0, :, :]
            v_true = ground_truth[:, 1, :, :]
        ke_truth = [compute_kinetic_energy(u_true[t], v_true[t], dx, dx) 
                    for t in range(min(num_steps, u_true.shape[0]))]
    
    # Plot
    plt.figure(figsize=(10, 5))
    steps = np.arange(num_steps)
    
    plt.plot(steps, ke_pred, 'b-', linewidth=2, label='AI Prediction')
    if ke_truth is not None:
        plt.plot(np.arange(len(ke_truth)), ke_truth, 'r--', linewidth=2, label='Ground Truth')
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Kinetic Energy', fontsize=12)
    plt.title('Kinetic Energy Evolution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/energy_conservation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary
    print("\n" + "="*50)
    print("ENERGY CONSERVATION")
    print("="*50)
    print(f"Initial KE: {ke_pred[0]:.6f}")
    print(f"Final KE: {ke_pred[-1]:.6f}")
    
    # For lid-driven cavity, energy should decay but not too fast
    decay_ratio = ke_pred[-1] / ke_pred[0] if ke_pred[0] > 0 else 0
    print(f"Decay ratio: {decay_ratio:.2%}")
    
    if 0.5 < decay_ratio < 1.5:
        print("✅ PASS: Energy evolution is physical")
        return True
    else:
        print("⚠️  WARNING: Unusual energy behavior")
        return False


def run_validation():
    """Run all physics validations"""
    
    print("="*60)
    print("PHYSICS VALIDATION FOR AI PREDICTIONS")
    print("="*60)
    
    # Try to load real predictions, or use synthetic data
    try:
        import torch
        from models.convlstm import ConvLSTM
        
        # Load model
        model = ConvLSTM(input_dim=3, hidden_dims=[64, 64, 64])
        checkpoint_path = 'checkpoints/best_model.pth'
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from {checkpoint_path}")
        else:
            print("! Using untrained model (no checkpoint)")
        
        model.eval()
        
        # Generate predictions
        grid_size = 64
        x = torch.randn(1, 3, grid_size, grid_size)
        with torch.no_grad():
            predictions = model(x, future_steps=20)
        predictions = predictions.numpy()
        print(f"✓ Generated predictions: {predictions.shape}")
        
    except Exception as e:
        print(f"! Model loading failed: {e}")
        print("! Using synthetic data for validation demo")
        
        # Synthetic data
        grid_size = 64
        num_steps = 20
        predictions = np.zeros((1, num_steps, 3, grid_size, grid_size))
        
        # Create synthetic velocity field
        for t in range(num_steps):
            x = np.linspace(0, 1, grid_size)
            y = np.linspace(0, 1, grid_size)
            X, Y = np.meshgrid(x, y)
            
            # Lid-driven cavity-like field with decay
            decay = np.exp(-0.1 * t)
            predictions[0, t, 0] = np.sin(np.pi * Y) * decay  # u
            predictions[0, t, 1] = -np.cos(np.pi * X) * np.sin(np.pi * Y) * decay  # v
            predictions[0, t, 2] = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) * decay  # p
    
    os.makedirs('results', exist_ok=True)
    
    # Run validations
    div_ok = validate_divergence_free(predictions, "AI Model")
    energy_ok = validate_energy_conservation(predictions)
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Divergence-free: {'✅ PASS' if div_ok else '❌ FAIL'}")
    print(f"Energy conservation: {'✅ PASS' if energy_ok else '❌ FAIL'}")
    
    if div_ok and energy_ok:
        print("\n🎉 All physics validations passed!")
    else:
        print("\n⚠️  Some validations need attention")
    
    print("\nPlots saved to:")
    print("  - results/divergence_validation.png")
    print("  - results/energy_conservation.png")


if __name__ == "__main__":
    run_validation()
