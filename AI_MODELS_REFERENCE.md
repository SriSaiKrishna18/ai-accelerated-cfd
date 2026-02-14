# AI Model Architectures: Quick Reference Guide

This document provides implementation templates for different AI architectures suitable for fluid flow prediction.

---

## 🎯 Model Selection Guide

| Model | Best For | Pros | Cons | Complexity |
|-------|----------|------|------|------------|
| **ConvLSTM** | Temporal sequences | Captures temporal dynamics well | Slow training | Medium |
| **U-Net** | Single-step prediction | Fast, accurate for spatial | No temporal memory | Low |
| **PINN** | Physics-constrained | Respects physical laws | Slow, harder to train | High |
| **Transformer** | Long sequences | Attention mechanism | Very slow, needs lots of data | Very High |
| **FNO** | Operator learning | Resolution-invariant | Complex implementation | High |

**Recommendation for this project**: Start with **ConvLSTM** (good balance of performance and complexity)

---

## 1. ConvLSTM (Recommended)

### Architecture Overview
```
Input (t=0): [B, 3, H, W]  (u, v, p)
   ↓
ConvLSTMCell × N layers
   ↓
Conv2d (1×1)
   ↓
Output (t=1...T): [B, T, 3, H, W]
```

### Full Implementation

```python
# models/convlstm.py
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell"""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM for sequence prediction"""
    def __init__(self, input_dim=3, hidden_dims=[64, 64, 64], 
                 kernel_size=3, num_layers=3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i-1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dims[i],
                kernel_size=kernel_size
            ))
        self.cell_list = nn.ModuleList(cell_list)
        
        self.conv_final = nn.Conv2d(
            in_channels=hidden_dims[-1],
            out_channels=input_dim,
            kernel_size=1
        )
    
    def forward(self, input_tensor, future_steps=1, hidden_state=None):
        """
        Args:
            input_tensor: [B, C, H, W] - initial state
            future_steps: number of steps to predict
            hidden_state: optional initial hidden state
        Returns:
            outputs: [B, T, C, H, W] - predicted sequence
        """
        b, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        layer_output_list = []
        last_state_list = []
        
        seq_len = future_steps
        cur_layer_input = input_tensor
        
        outputs = []
        
        for t in range(seq_len):
            # Forward through all layers
            h_list = []
            c_list = []
            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                h, c = self.cell_list[layer_idx](cur_layer_input, (h, c))
                h_list.append(h)
                c_list.append(c)
                cur_layer_input = h
            
            hidden_state = list(zip(h_list, c_list))
            
            # Output
            output = self.conv_final(h_list[-1])
            outputs.append(output)
            
            # Use prediction as next input (autoregressive)
            cur_layer_input = output
        
        outputs = torch.stack(outputs, dim=1)  # [B, T, C, H, W]
        return outputs
    
    def _init_hidden(self, batch_size, image_size):
        height, width = image_size
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                (torch.zeros(batch_size, self.hidden_dims[i], height, width, device=self.conv_final.weight.device),
                 torch.zeros(batch_size, self.hidden_dims[i], height, width, device=self.conv_final.weight.device))
            )
        return init_states

# Example usage
if __name__ == "__main__":
    model = ConvLSTM(input_dim=3, hidden_dims=[64, 64, 64], num_layers=3)
    x = torch.randn(4, 3, 128, 128)  # [B, C, H, W]
    out = model(x, future_steps=10)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # [4, 10, 3, 128, 128]
```

### Training Script

```python
# train_convlstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.convlstm import ConvLSTM
from data.dataset import FluidDataset
import wandb

def train_epoch(model, dataloader, optimizer, criterion, device):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            wandb.log({'batch_loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
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

def main():
    # Config
    config = {
        'batch_size': 8,
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'hidden_dims': [64, 64, 64],
        'num_layers': 3,
    }
    
    wandb.init(project='fluid-convlstm', config=config)
    
    # Data
    train_dataset = FluidDataset('data/training.h5', seq_len=5, pred_len=10)
    val_dataset = FluidDataset('data/validation.h5', seq_len=5, pred_len=10)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=4)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvLSTM(
        input_dim=3,
        hidden_dims=config['hidden_dims'],
        num_layers=config['num_layers']
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'checkpoints/best_model.pth')
    
    wandb.finish()

if __name__ == "__main__":
    main()
```

---

## 2. U-Net

### Architecture
```
     Encoder              Decoder
Input → Conv → Pool → ... → Upsample → Conv → Output
         ↓               ↗ (skip connection)
       Feature maps
```

### Implementation

```python
# models/unet.py
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, 2))
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)

# Multi-step prediction wrapper
class UNetSequential(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.unet = UNet(in_channels, out_channels)
    
    def forward(self, x, future_steps=1):
        outputs = []
        current = x
        
        for _ in range(future_steps):
            pred = self.unet(current)
            outputs.append(pred)
            current = pred  # Autoregressive
        
        return torch.stack(outputs, dim=1)
```

---

## 3. Physics-Informed Neural Network (PINN)

### Implementation

```python
# models/pinn.py
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers=[3, 128, 128, 128, 3]):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = nn.Tanh()
    
    def forward(self, x, y, t):
        """
        Args:
            x, y: spatial coordinates [B, 1]
            t: time [B, 1]
        Returns:
            u, v, p: predicted fields [B, 1] each
        """
        inputs = torch.cat([x, y, t], dim=1)
        
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        
        outputs = self.layers[-1](inputs)
        u, v, p = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
        
        return u, v, p
    
    def physics_loss(self, x, y, t, nu=0.01):
        """Compute physics-based loss from Navier-Stokes equations"""
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        
        u, v, p = self.forward(x, y, t)
        
        # First derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        
        # Navier-Stokes residuals
        f_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        f_c = u_x + v_y  # Continuity
        
        loss = torch.mean(f_u**2 + f_v**2 + f_c**2)
        return loss

# Training with physics loss
def train_pinn(model, data_points, physics_points, nu=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(10000):
        optimizer.zero_grad()
        
        # Data loss
        x_data, y_data, t_data, u_data, v_data, p_data = data_points
        u_pred, v_pred, p_pred = model(x_data, y_data, t_data)
        
        data_loss = nn.MSELoss()(u_pred, u_data) + \
                    nn.MSELoss()(v_pred, v_data) + \
                    nn.MSELoss()(p_pred, p_data)
        
        # Physics loss
        x_phys, y_phys, t_phys = physics_points
        physics_loss = model.physics_loss(x_phys, y_phys, t_phys, nu)
        
        # Combined loss
        loss = data_loss + 0.1 * physics_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
```

---

## 📊 Performance Comparison

| Model | Training Time | Inference Time | Accuracy | Memory |
|-------|---------------|----------------|----------|--------|
| ConvLSTM | 2-4 hours | 10ms/frame | High | Medium |
| U-Net | 1-2 hours | 5ms/frame | Medium-High | Low |
| PINN | 8-12 hours | 50ms/frame | Medium | Low |

---

## 🎓 Tips for Success

1. **Start Simple**: Begin with U-Net for single-step, then move to ConvLSTM
2. **Data Quality**: More important than model complexity
3. **Regularization**: Use dropout, batch norm to prevent overfitting
4. **Learning Rate**: Start with 1e-3, reduce by 0.5 when plateau
5. **Validation**: Always validate on unseen initial conditions

---

## 🔧 Troubleshooting

**Problem**: Model predicts all zeros  
**Solution**: Check data normalization, reduce learning rate

**Problem**: Training diverges  
**Solution**: Clip gradients, reduce batch size, check CFL in data

**Problem**: Good training loss, poor validation  
**Solution**: Add dropout, more data variation, early stopping

**Problem**: Slow convergence  
**Solution**: Use Adam optimizer, learning rate scheduling, mixed precision

