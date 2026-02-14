"""
Training Script for Fluid Flow AI Model
AI-HPC Hybrid Project

Trains ConvLSTM model to predict future fluid states from current state.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.convlstm import ConvLSTM, UNetSequential
from data.dataset import NumpyDataset, create_dummy_data


def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)   # [B, T_in, C, H, W]
        targets = targets.to(device) # [B, T_out, C, H, W]
        
        # Use last input frame as initial state
        initial_state = inputs[:, -1]  # [B, C, H, W]
        
        # Predict future
        predictions = model(initial_state, future_steps=targets.shape[1])
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            initial_state = inputs[:, -1]
            predictions = model(initial_state, future_steps=targets.shape[1])
            
            loss = criterion(predictions, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def compute_metrics(predictions, targets):
    """Compute evaluation metrics."""
    with torch.no_grad():
        mse = nn.MSELoss()(predictions, targets).item()
        rmse = np.sqrt(mse)
        mae = nn.L1Loss()(predictions, targets).item()
        max_err = torch.max(torch.abs(predictions - targets)).item()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_err
    }


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path):
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def main(args):
    """Main training function."""
    print("=" * 50)
    print("Fluid Flow AI Model Training")
    print("=" * 50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dummy data if needed
    train_data_path = os.path.join(args.data_dir, 'train_data.npz')
    val_data_path = os.path.join(args.data_dir, 'val_data.npz')
    
    if not os.path.exists(train_data_path):
        print("Creating training data...")
        create_dummy_data(train_data_path, num_frames=200, nx=args.grid_size, ny=args.grid_size)
    
    if not os.path.exists(val_data_path):
        print("Creating validation data...")
        create_dummy_data(val_data_path, num_frames=50, nx=args.grid_size, ny=args.grid_size)
    
    # Datasets
    print("Loading datasets...")
    train_dataset = NumpyDataset(train_data_path, seq_len=args.seq_len, pred_len=args.pred_len)
    val_dataset = NumpyDataset(val_data_path, seq_len=args.seq_len, pred_len=args.pred_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Model
    print("Creating model...")
    if args.model == 'convlstm':
        model = ConvLSTM(
            input_dim=3,
            hidden_dims=[args.hidden_dim] * args.num_layers,
            num_layers=args.num_layers
        ).to(device)
    else:
        model = UNetSequential(in_channels=3, out_channels=3).to(device)
    
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    print("-" * 50)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, 
                          os.path.join(args.output_dir, 'best_model.pth'))
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                          os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'))
    
    print("-" * 50)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, val_loss,
                   os.path.join(args.output_dir, 'final_model.pth'))
    
    # Save training history
    np.savez(os.path.join(args.output_dir, 'history.npz'),
             train_loss=history['train_loss'],
             val_loss=history['val_loss'])
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            initial_state = inputs[:, -1]
            predictions = model(initial_state, future_steps=targets.shape[1])
            
            all_preds.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(all_preds, all_targets)
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  Max Error: {metrics['max_error']:.6f}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fluid flow AI model")
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/training',
                        help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    # Model
    parser.add_argument('--model', type=str, default='convlstm',
                        choices=['convlstm', 'unet'],
                        help='Model architecture')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension for ConvLSTM')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of ConvLSTM layers')
    parser.add_argument('--grid-size', type=int, default=64,
                        help='Grid size (nx=ny)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seq-len', type=int, default=5,
                        help='Input sequence length')
    parser.add_argument('--pred-len', type=int, default=10,
                        help='Prediction sequence length')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)
