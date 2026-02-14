"""
Visualization and Validation Script
AI-HPC Hybrid Project

Creates accuracy comparison plots and validation report.
"""

import os
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.convlstm import ConvLSTM
from data.dataset import NumpyDataset


def load_checkpoint_file(filepath):
    """Load HPC checkpoint file."""
    with open(filepath, 'rb') as f:
        nx = struct.unpack('i', f.read(4))[0]
        ny = struct.unpack('i', f.read(4))[0]
        time = struct.unpack('d', f.read(8))[0]
        step = struct.unpack('i', f.read(4))[0]
        
        n = nx * ny
        u = np.array(struct.unpack(f'{n}d', f.read(n * 8))).reshape((ny, nx))
        v = np.array(struct.unpack(f'{n}d', f.read(n * 8))).reshape((ny, nx))
        p = np.array(struct.unpack(f'{n}d', f.read(n * 8))).reshape((ny, nx))
        
    return {'u': u, 'v': v, 'p': p, 'time': time, 'step': step, 'nx': nx, 'ny': ny}


def compute_metrics(pred, true):
    """Compute error metrics."""
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - true))
    max_err = np.max(np.abs(pred - true))
    rel_err = rmse / (np.max(np.abs(true)) + 1e-10)
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'max_error': max_err, 'relative_error': rel_err}


def create_comparison_plots(model_path, data_path, output_dir='results'):
    """Create accuracy comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model and data...")
    device = torch.device('cpu')
    
    # Load model
    model = ConvLSTM(input_dim=3, hidden_dims=[64, 64, 64]).to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load validation data
    dataset = NumpyDataset(data_path, seq_len=5, pred_len=10)
    
    # Get a sample
    input_seq, target_seq = dataset[0]
    
    # Predict
    with torch.no_grad():
        initial_state = input_seq[-1].unsqueeze(0)  # [1, 3, H, W]
        predictions = model(initial_state, future_steps=10)
    
    predictions = predictions[0].numpy()  # [10, 3, H, W]
    targets = target_seq.numpy()  # [10, 3, H, W]
    
    # Compute per-step errors
    rmse_u = []
    rmse_v = []
    rmse_p = []
    
    for t in range(10):
        rmse_u.append(np.sqrt(np.mean((predictions[t, 0] - targets[t, 0])**2)))
        rmse_v.append(np.sqrt(np.mean((predictions[t, 1] - targets[t, 1])**2)))
        rmse_p.append(np.sqrt(np.mean((predictions[t, 2] - targets[t, 2])**2)))
    
    # Plot 1: Error vs Time Step
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = range(1, 11)
    ax.plot(steps, rmse_u, 'b-o', label='u-velocity', linewidth=2)
    ax.plot(steps, rmse_v, 'r-s', label='v-velocity', linewidth=2)
    ax.plot(steps, rmse_p, 'g-^', label='pressure', linewidth=2)
    ax.set_xlabel('Prediction Step', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('AI Prediction Error vs Time Step', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_vs_timestep.png'), dpi=150)
    print(f"Saved: {output_dir}/error_vs_timestep.png")
    
    # Plot 2: Field Comparison at step 5
    step = 4  # 0-indexed
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    
    fields = ['u-velocity', 'v-velocity', 'pressure']
    for i, name in enumerate(fields):
        pred = predictions[step, i]
        true = targets[step, i]
        diff = pred - true
        
        vmin, vmax = true.min(), true.max()
        
        im1 = axes[i, 0].imshow(true, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'{name} (Ground Truth)')
        plt.colorbar(im1, ax=axes[i, 0])
        
        im2 = axes[i, 1].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'{name} (AI Prediction)')
        plt.colorbar(im2, ax=axes[i, 1])
        
        diff_max = max(abs(diff.min()), abs(diff.max()))
        im3 = axes[i, 2].imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        axes[i, 2].set_title(f'{name} (Error)')
        plt.colorbar(im3, ax=axes[i, 2])
    
    plt.suptitle(f'HPC vs AI Comparison (Step {step+1})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'field_comparison.png'), dpi=150)
    print(f"Saved: {output_dir}/field_comparison.png")
    
    # Plot 3: Training history if available
    history_path = os.path.join(os.path.dirname(model_path), 'history.npz')
    if os.path.exists(history_path):
        history = np.load(history_path)
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.set_title('Training History', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
        print(f"Saved: {output_dir}/training_history.png")
    
    # Compute overall metrics
    overall_metrics = {
        'u': compute_metrics(predictions[:, 0], targets[:, 0]),
        'v': compute_metrics(predictions[:, 1], targets[:, 1]),
        'p': compute_metrics(predictions[:, 2], targets[:, 2])
    }
    
    return overall_metrics


def generate_validation_report(metrics, output_dir='results'):
    """Generate detailed validation report."""
    report_path = os.path.join(output_dir, 'validation_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# AI Model Validation Report\n\n")
        f.write("## Navier-Stokes 2D AI-HPC Hybrid Project\n\n")
        f.write("---\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Field | RMSE | MAE | Max Error | Relative Error |\n")
        f.write("|-------|------|-----|-----------|----------------|\n")
        for field, m in metrics.items():
            f.write(f"| {field} | {m['rmse']:.6f} | {m['mae']:.6f} | {m['max_error']:.6f} | {m['relative_error']*100:.2f}% |\n")
        f.write("\n")
        
        # Combined RMSE
        combined_rmse = np.sqrt(metrics['u']['mse'] + metrics['v']['mse'] + metrics['p']['mse'])
        f.write(f"**Combined RMSE**: {combined_rmse:.6f}\n\n")
        
        f.write("---\n\n")
        f.write("## Plots\n\n")
        f.write("### Error vs Prediction Step\n")
        f.write("![Error vs Time](error_vs_timestep.png)\n\n")
        f.write("### Field Comparison\n")
        f.write("![Field Comparison](field_comparison.png)\n\n")
        f.write("### Training History\n")
        f.write("![Training History](training_history.png)\n\n")
        
        f.write("---\n\n")
        f.write("## Conclusions\n\n")
        if combined_rmse < 0.1:
            f.write("[EXCELLENT]: AI predictions closely match HPC ground truth.\n")
        elif combined_rmse < 0.2:
            f.write("[GOOD]: AI predictions are acceptably accurate.\n")
        else:
            f.write("[NEEDS IMPROVEMENT]: Consider more training data or epochs.\n")
        
        f.write("\n## Reproducibility\n\n")
        f.write("```bash\n")
        f.write("# Generate data\n")
        f.write("python python/generate_data.py --mode synthetic --num-trajectories 20\n\n")
        f.write("# Train model\n")
        f.write("python python/training/train.py --epochs 50\n\n")
        f.write("# Create plots\n")
        f.write("python python/visualize.py\n")
        f.write("```\n")
    
    print(f"Saved: {report_path}")
    return report_path


if __name__ == "__main__":
    model_path = 'checkpoints/best_model.pth'
    data_path = 'data/training/val_data.npz'
    output_dir = 'results'
    
    print("Creating comparison plots...")
    metrics = create_comparison_plots(model_path, data_path, output_dir)
    
    print("\nGenerating validation report...")
    generate_validation_report(metrics, output_dir)
    
    print("\n=== Validation Complete ===")
    print(f"Results saved to: {output_dir}/")
