"""
Hybrid HPC-AI Solver
AI-HPC Hybrid Project

Integrates HPC Navier-Stokes solver with AI prediction model.
Workflow: HPC → Checkpoint → AI Prediction → Validation
"""

import os
import sys
import struct
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.convlstm import ConvLSTM


class HybridFluidSolver:
    """
    Hybrid solver combining HPC computation with AI prediction.
    
    Workflow:
    1. Run HPC solver to checkpoint time
    2. Load AI model
    3. Predict future states using AI
    4. Optionally validate against full HPC computation
    """
    
    def __init__(self, model_path=None, nx=128, ny=128):
        """
        Args:
            model_path: Path to trained AI model checkpoint
            nx, ny: Grid dimensions
        """
        self.nx = nx
        self.ny = ny
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load AI model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("Warning: No model loaded. Using default ConvLSTM.")
            self.model = ConvLSTM(input_dim=3, hidden_dims=[64, 64, 64]).to(self.device)
    
    def load_model(self, model_path):
        """Load trained AI model from checkpoint."""
        print(f"Loading model from {model_path}...")
        
        self.model = ConvLSTM(input_dim=3, hidden_dims=[64, 64, 64]).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully.")
    
    def load_checkpoint(self, checkpoint_path):
        """Load HPC checkpoint file."""
        with open(checkpoint_path, 'rb') as f:
            nx = struct.unpack('i', f.read(4))[0]
            ny = struct.unpack('i', f.read(4))[0]
            time = struct.unpack('d', f.read(8))[0]
            step = struct.unpack('i', f.read(4))[0]
            
            n = nx * ny
            u = np.array(struct.unpack(f'{n}d', f.read(n * 8)))
            v = np.array(struct.unpack(f'{n}d', f.read(n * 8)))
            p = np.array(struct.unpack(f'{n}d', f.read(n * 8)))
            
            u = u.reshape((ny, nx))
            v = v.reshape((ny, nx))
            p = p.reshape((ny, nx))
        
        return {
            'u': u, 'v': v, 'p': p,
            'time': time, 'step': step,
            'nx': nx, 'ny': ny
        }
    
    def predict(self, u, v, p, num_steps=10):
        """
        Predict future states using AI model.
        
        Args:
            u, v, p: Current velocity and pressure fields [H, W]
            num_steps: Number of future steps to predict
        
        Returns:
            predictions: dict with 'u', 'v', 'p' arrays of shape [T, H, W]
        """
        if self.model is None:
            raise RuntimeError("No model loaded!")
        
        # Prepare input: [B, C, H, W]
        state = np.stack([u, v, p], axis=0).astype(np.float32)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(state_tensor, future_steps=num_steps)
        
        # Convert back to numpy
        predictions = predictions.cpu().numpy()[0]  # [T, C, H, W]
        
        return {
            'u': predictions[:, 0],  # [T, H, W]
            'v': predictions[:, 1],
            'p': predictions[:, 2]
        }
    
    def validate(self, predicted, ground_truth):
        """
        Compare AI predictions against HPC ground truth.
        
        Args:
            predicted: dict with 'u', 'v', 'p' arrays [T, H, W]
            ground_truth: dict with 'u', 'v', 'p' arrays [T, H, W]
        
        Returns:
            metrics: dict with error metrics
        """
        metrics = {}
        
        for field in ['u', 'v', 'p']:
            pred = predicted[field]
            true = ground_truth[field]
            
            # Compute errors
            mse = np.mean((pred - true) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred - true))
            max_err = np.max(np.abs(pred - true))
            
            # Relative error
            true_max = np.max(np.abs(true)) + 1e-10
            rel_err = rmse / true_max
            
            metrics[field] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'max_error': max_err,
                'relative_error': rel_err
            }
        
        # Combined metrics
        combined_rmse = np.sqrt(
            metrics['u']['mse'] + metrics['v']['mse'] + metrics['p']['mse']
        )
        metrics['combined_rmse'] = combined_rmse
        
        return metrics
    
    def visualize_comparison(self, predicted, ground_truth, step=0, save_path=None):
        """
        Create comparison visualization of AI vs HPC.
        
        Args:
            predicted: dict with 'u', 'v', 'p' arrays
            ground_truth: dict with 'u', 'v', 'p' arrays
            step: Which time step to visualize
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        fields = ['u', 'v', 'p']
        field_names = ['u-velocity', 'v-velocity', 'pressure']
        
        for i, (field, name) in enumerate(zip(fields, field_names)):
            pred = predicted[field][step] if predicted[field].ndim == 3 else predicted[field]
            true = ground_truth[field][step] if ground_truth[field].ndim == 3 else ground_truth[field]
            diff = pred - true
            
            vmin, vmax = true.min(), true.max()
            
            # Ground truth
            im1 = axes[i, 0].imshow(true, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f'{name} (HPC)')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Prediction
            im2 = axes[i, 1].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f'{name} (AI)')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Difference
            diff_max = max(abs(diff.min()), abs(diff.max()))
            im3 = axes[i, 2].imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
            axes[i, 2].set_title(f'{name} (Error)')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.suptitle(f'HPC vs AI Comparison (Step {step})', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def run_hybrid_simulation(self, checkpoint_path, num_predict_steps=10):
        """
        Run complete hybrid simulation workflow.
        
        Args:
            checkpoint_path: Path to HPC checkpoint file
            num_predict_steps: Number of steps to predict with AI
        
        Returns:
            predictions: AI-predicted future states
            checkpoint: Loaded checkpoint data
        """
        print("=" * 50)
        print("Hybrid HPC-AI Simulation")
        print("=" * 50)
        
        # Load checkpoint
        print(f"\n1. Loading HPC checkpoint: {checkpoint_path}")
        checkpoint = self.load_checkpoint(checkpoint_path)
        print(f"   Grid: {checkpoint['nx']} x {checkpoint['ny']}")
        print(f"   Time: {checkpoint['time']:.4f}")
        print(f"   Step: {checkpoint['step']}")
        
        # Predict with AI
        print(f"\n2. Running AI prediction for {num_predict_steps} steps...")
        predictions = self.predict(
            checkpoint['u'], checkpoint['v'], checkpoint['p'],
            num_steps=num_predict_steps
        )
        print(f"   Prediction shape: ({num_predict_steps}, {checkpoint['ny']}, {checkpoint['nx']})")
        
        print("\n" + "=" * 50)
        print("Hybrid simulation complete!")
        
        return predictions, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run hybrid HPC-AI simulation")
    
    parser.add_argument('--checkpoint', type=str, default='data/checkpoints/checkpoint.bin',
                       help='Path to HPC checkpoint file')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained AI model')
    parser.add_argument('--predict-steps', type=int, default=10,
                       help='Number of steps to predict')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create solver
    solver = HybridFluidSolver(model_path=args.model)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please run the C++ solver first to generate a checkpoint.")
        return
    
    # Run hybrid simulation
    predictions, checkpoint = solver.run_hybrid_simulation(
        args.checkpoint,
        num_predict_steps=args.predict_steps
    )
    
    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'ai_predictions.npz')
    np.savez(output_path,
             u=predictions['u'],
             v=predictions['v'],
             p=predictions['p'],
             checkpoint_time=checkpoint['time'])
    print(f"\nPredictions saved to {output_path}")
    
    # Visualize if requested
    if args.visualize:
        # Use checkpoint as "ground truth" (first frame comparison)
        gt = {
            'u': np.expand_dims(checkpoint['u'], 0),
            'v': np.expand_dims(checkpoint['v'], 0),
            'p': np.expand_dims(checkpoint['p'], 0)
        }
        
        vis_path = os.path.join(args.output_dir, 'prediction_comparison.png')
        solver.visualize_comparison(predictions, gt, step=0, save_path=vis_path)


if __name__ == "__main__":
    main()
