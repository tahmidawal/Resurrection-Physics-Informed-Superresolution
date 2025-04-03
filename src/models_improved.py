import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Basic convolutional block with double convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class ContextAwareUNet(nn.Module):
    def __init__(self, in_channels: int = 2, context_padding: int = 2):
        """
        Enhanced U-Net architecture for PDE solution upscaling with context awareness.
        Specifically optimized for 24x24 -> 48x48 upscaling.
        
        Args:
            in_channels: Number of input channels (coarse solution + f)
            context_padding: Padding size on each side for context window
        """
        super().__init__()
        
        self.context_padding = context_padding
        
        # Encoder (moderate depth)
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        
        # Bridge with dilated convolutions for larger receptive field
        self.bridge = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Decoder with skip connections
        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)
        
        # Multi-scale output
        self.out_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.out_bn1 = nn.BatchNorm2d(32)
        self.out_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.out_bn2 = nn.BatchNorm2d(16)
        self.final = nn.Conv2d(16, 1, kernel_size=1)
        
        # Attention gates
        self.att3 = AttentionGate(256, 512)
        self.att2 = AttentionGate(128, 256)
        self.att1 = AttentionGate(64, 128)
        
        # Max pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Boundary attention module
        self.boundary_attention = BoundaryAttentionModule(64, 32)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Split input channels
        coarse_solution = x[:, 0:1, :, :]
        features = x[:, 1:, :, :]  # Now only contains f
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bridge
        b = self.bridge(e3)
        
        # Decoder with attention and skip connections
        e3_att = self.att3(e3, b)
        d3 = self.dec3(torch.cat([b, e3_att], dim=1))
        
        e2_att = self.att2(e2, self.up(d3))
        d2 = self.dec2(torch.cat([self.up(d3), e2_att], dim=1))
        
        e1_att = self.att1(e1, self.up(d2))
        d1 = self.dec1(torch.cat([self.up(d2), e1_att], dim=1))
        
        # Apply boundary attention to focus on edge quality
        d1 = self.boundary_attention(d1)
        
        # Multi-scale refinement
        x = F.relu(self.out_bn1(self.out_conv1(d1)))
        x = F.relu(self.out_bn2(self.out_conv2(x)))
        x = self.final(x)
        
        # Residual connection from coarse solution
        x = x + coarse_solution
        
        # Extract the core region (removing context padding)
        if self.context_padding > 0:
            # Calculate dimensions for the core region
            h, w = x.shape[2], x.shape[3]
            
            # Ensure the output size is exactly 48x48 for the 24x24 -> 48x48 upscaling
            core_h = 48
            core_w = 48
            
            # Calculate appropriate padding to extract if the tensor is larger
            # This handles the case with context padding
            if h > core_h:
                pad_h = (h - core_h) // 2
                pad_w = (w - core_w) // 2
                x = x[:, :, pad_h:pad_h+core_h, pad_w:pad_w+core_w]
        
        return x

class AttentionGate(nn.Module):
    def __init__(self, in_channels: int, gating_channels: int, reduction: int = 8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(gating_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention from gating signal
        if gating.shape[-2:] != x.shape[-2:]:
            gating = F.interpolate(gating, size=x.shape[-2:], mode='bilinear', align_corners=True)
        sa = self.spatial_attention(gating)
        x = x * sa
        
        return x

class BoundaryAttentionModule(nn.Module):
    def __init__(self, in_channels: int, intermediate_channels: int):
        """
        Module that pays special attention to boundary regions.
        
        Args:
            in_channels: Number of input channels
            intermediate_channels: Number of intermediate channels
        """
        super().__init__()
        
        # Edge detection kernels (Sobel filters)
        self.edge_conv = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1, bias=False)
        
        # Initialize with modified Sobel filters for edge detection
        sobel_kernel = torch.zeros(intermediate_channels, in_channels, 3, 3)
        base_kernel = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)
        
        # Create kernels for each input/output channel combination
        for i in range(intermediate_channels):
            channel_idx = i % in_channels
            sobel_kernel[i, channel_idx] = base_kernel
        
        self.edge_conv.weight = nn.Parameter(sobel_kernel)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Detect edges
        edge_features = self.edge_conv(x)
        edge_map = torch.mean(torch.abs(edge_features), dim=1, keepdim=True)
        
        # Create attention weights
        attention = self.attention(edge_features)
        
        # Apply attention
        x = torch.cat([x, attention], dim=1)
        x = self.refine(x)
        
        return x

class OverlappingPDEDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: dict, device: str = 'cuda', core_size: Tuple[int, int] = (20, 40), context_size: Tuple[int, int] = (24, 48)):
        """
        Dataset class for PDE solutions with overlapping context windows.
        Optimized for 24x24 -> 48x48 upscaling.
        
        Args:
            data_dict: Dictionary containing the dataset
            device: Device to store tensors on
            core_size: Size of core region (coarse, fine)
            context_size: Size of context region (coarse, fine)
        """
        self.device = device
        self.core_size = core_size
        self.context_size = context_size
        
        # Calculate padding size
        self.coarse_padding = (context_size[0] - core_size[0]) // 2
        self.fine_padding = (context_size[1] - core_size[1]) // 2
        
        # Convert numpy arrays to tensors and move to device
        self.u_coarse = torch.from_numpy(data_dict['u_coarse']).float().to(device)  # Shape: [N, 24, 24]
        self.u_fine = torch.from_numpy(data_dict['u_fine']).float().to(device)      # Shape: [N, 48, 48]
        self.f_fine = torch.from_numpy(data_dict['f_fine']).float().to(device)      # Shape: [N, 48, 48]
        
        # Compute normalization statistics
        self.u_mean = self.u_fine.mean()
        self.u_std = self.u_fine.std()
        self.f_mean = self.f_fine.mean()
        self.f_std = self.f_fine.std()
        
        # Normalize the data
        self.u_fine_norm = (self.u_fine - self.u_mean) / self.u_std
        self.u_coarse_norm = (self.u_coarse - self.u_mean) / self.u_std
        self.f_fine_norm = (self.f_fine - self.f_mean) / self.f_std
        
        # Upsample coarse solution to fine grid (24x24 -> 48x48)
        self.u_coarse_upsampled = F.interpolate(
            self.u_coarse_norm.unsqueeze(1),
            size=(48, 48),  # Upscale to 48x48
            mode='bilinear',
            align_corners=True
        )
        
    def __len__(self) -> int:
        return len(self.u_fine)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Combine normalized inputs: [upsampled_coarse_solution, f]
        x = torch.cat([
            self.u_coarse_upsampled[idx],
            self.f_fine_norm[idx].unsqueeze(0)
        ], dim=0)
        
        # For 24x24 -> 48x48 upscaling, we use the full fine solution
        y = self.u_fine_norm[idx].unsqueeze(0)
        
        return x, y
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize the model output back to original scale."""
        return x * self.u_std + self.u_mean
    
    @staticmethod
    def stitch_predictions(predictions: List[torch.Tensor], grid_size: int, overlap: int = 4) -> torch.Tensor:
        """
        Stitch together predictions from overlapping subdomains.
        
        Args:
            predictions: List of predictions (each corresponding to one subdomain)
            grid_size: Number of subdomains per dimension
            overlap: Number of pixels to overlap when stitching
            
        Returns:
            Stitched solution
        """
        # Determine dimensions of one prediction
        core_h, core_w = predictions[0].shape[1], predictions[0].shape[2]
        
        # Determine dimensions of full grid
        full_h = core_h * grid_size - overlap * (grid_size - 1)
        full_w = core_w * grid_size - overlap * (grid_size - 1)
        
        # Initialize output tensor
        stitched = torch.zeros((1, full_h, full_w), device=predictions[0].device)
        
        # Initialize weight tensor for averaging overlapping regions
        weights = torch.zeros((1, full_h, full_w), device=predictions[0].device)
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate starting position
                start_h = i * (core_h - overlap)
                start_w = j * (core_w - overlap)
                
                # Create linear weight mask for blending (higher weights for central regions)
                weight_mask = torch.ones_like(predictions[idx])
                
                # Apply prediction
                stitched[:, start_h:start_h+core_h, start_w:start_w+core_w] += predictions[idx] * weight_mask
                weights[:, start_h:start_h+core_h, start_w:start_w+core_w] += weight_mask
                
                idx += 1
        
        # Average overlapping regions
        stitched = stitched / (weights + 1e-8)
        
        return stitched

def init_weights(m: nn.Module):
    """
    Initialize network weights using Kaiming initialization.
    
    Args:
        m: PyTorch module
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class BoundaryAwareLoss(nn.Module):
    def __init__(self, lambda_boundary: float = 2.0):
        """
        Loss function that gives higher weight to boundaries/edges.
        
        Args:
            lambda_boundary: Weight multiplier for boundary regions
        """
        super().__init__()
        self.lambda_boundary = lambda_boundary
        self.mse = nn.MSELoss()
        
        # Define Sobel filters for edge detection
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Move Sobel filters to the same device as input tensors
        device = prediction.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        
        # Calculate gradients (edges) in target image
        # Pad to maintain input size
        padded_target = F.pad(target, (1, 1, 1, 1), mode='reflect')
        grad_x = F.conv2d(padded_target, sobel_x)
        grad_y = F.conv2d(padded_target, sobel_y)
        target_grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Create a weight map that emphasizes edges
        edge_weights = 1.0 + self.lambda_boundary * (target_grad_magnitude / (target_grad_magnitude.max() + 1e-8))
        
        # Compute weighted MSE loss
        squared_diff = (prediction - target)**2
        weighted_squared_diff = squared_diff * edge_weights
        
        # Return mean loss
        return weighted_squared_diff.mean() 