"""
ConvLSTM Model for Fluid Flow Prediction
AI-HPC Hybrid Project

Predicts future fluid states (u, v, p) from current state using
Convolutional LSTM architecture for temporal sequence modeling.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell.
    
    Replaces matrix multiplications with convolutions to preserve
    spatial structure in the hidden state.
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Combined convolution for all gates (input, forget, output, cell)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        """
        Forward pass through ConvLSTM cell.
        
        Args:
            input_tensor: [B, C, H, W] - input at current timestep
            cur_state: tuple of (h, c) hidden and cell states
        
        Returns:
            h_next: next hidden state
            (h_next, c_next): tuple for next hidden state storage
        """
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Gate activations
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate
        
        # Update cell and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        # Return h_next and tuple for state storage (matches Kaggle model)
        return h_next, (h_next, c_next)


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM for fluid state prediction.
    
    Takes initial state (u, v, p) and predicts future states
    in an autoregressive manner.
    
    NOTE: This architecture matches the Kaggle-trained model.
    """
    
    def __init__(self, input_dim=3, hidden_dims=None, kernel_size=3, num_layers=3):
        """
        Args:
            input_dim: Number of input channels (3 for u, v, p)
            hidden_dims: List of hidden dimensions for each layer
            kernel_size: Convolution kernel size
            num_layers: Number of ConvLSTM layers (ignored, uses len(hidden_dims))
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        
        # Input projection (matches Kaggle model)
        self.input_conv = nn.Conv2d(input_dim, hidden_dims[0], 3, padding=1)
        
        # ConvLSTM layers (matches Kaggle model with self.cells)
        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = hidden_dims[i]
            self.cells.append(ConvLSTMCell(
                input_dim=in_dim,
                hidden_dim=hidden_dims[i],
                kernel_size=kernel_size
            ))
        
        # Output projection (matches Kaggle model)
        self.output_conv = nn.Conv2d(hidden_dims[-1], input_dim, 1)
    
    def forward(self, input_tensor, future_steps=1, hidden_state=None):
        """
        Forward pass predicting future_steps timesteps.
        
        Args:
            input_tensor: [B, C, H, W] - initial state
            future_steps: Number of future steps to predict
            hidden_state: Optional initial hidden state
        
        Returns:
            outputs: [B, T, C, H, W] - predicted sequence
        """
        b, _, h, w = input_tensor.size()
        device = input_tensor.device
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = [self._init_hidden_layer(b, h, w, self.hidden_dims[i], device) 
                          for i in range(self.num_layers)]
        
        outputs = []
        current_input = input_tensor
        
        for t in range(future_steps):
            # Input projection
            h_state = self.input_conv(current_input)
            
            # Forward through all layers
            for layer_idx, cell in enumerate(self.cells):
                h_state, hidden_state[layer_idx] = cell(h_state, hidden_state[layer_idx])
            
            # Output projection
            output = self.output_conv(h_state)
            outputs.append(output)
            
            # Autoregressive: use prediction as next input
            current_input = output
        
        outputs = torch.stack(outputs, dim=1)  # [B, T, C, H, W]
        return outputs
    
    def _init_hidden_layer(self, batch_size, height, width, hidden_dim, device):
        """Initialize hidden states for one layer."""
        h = torch.zeros(batch_size, hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, hidden_dim, height, width, device=device)
        return (h, c)
    
    def _init_hidden(self, batch_size, image_size):
        """Initialize hidden states to zeros (legacy method for compatibility)."""
        height, width = image_size
        device = self.output_conv.weight.device
        return [self._init_hidden_layer(batch_size, height, width, self.hidden_dims[i], device) 
                for i in range(self.num_layers)]


class UNet(nn.Module):
    """
    U-Net architecture for single-step fluid state prediction.
    
    Encoder-decoder with skip connections for preserving spatial details.
    """
    
    def __init__(self, in_channels=3, out_channels=3, features=None):
        super().__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder
        for feature in features:
            self.downs.append(self._double_conv(in_channels, feature))
            in_channels = feature
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, 2))
            self.ups.append(self._double_conv(feature*2, feature))
        
        self.bottleneck = self._double_conv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx+1](x)
        
        return self.final_conv(x)


class UNetSequential(nn.Module):
    """
    U-Net wrapper for multi-step autoregressive prediction.
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.unet = UNet(in_channels, out_channels)
    
    def forward(self, x, future_steps=1):
        outputs = []
        current = x
        
        for _ in range(future_steps):
            pred = self.unet(current)
            outputs.append(pred)
            current = pred
        
        return torch.stack(outputs, dim=1)


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == "__main__":
    print("Testing ConvLSTM model...")
    
    # Test ConvLSTM
    model = ConvLSTM(input_dim=3, hidden_dims=[64, 64, 64], num_layers=3)
    x = torch.randn(4, 3, 128, 128)  # [B, C, H, W]
    out = model(x, future_steps=10)
    
    print(f"ConvLSTM:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")  # [4, 10, 3, 128, 128]
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test U-Net
    unet = UNetSequential(in_channels=3, out_channels=3)
    out_unet = unet(x, future_steps=5)
    
    print(f"\nU-Net Sequential:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_unet.shape}")  # [4, 5, 3, 128, 128]
    print(f"  Parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    print("\nModel tests passed!")
