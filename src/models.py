import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

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

class UNet(nn.Module):
    def __init__(self, in_channels: int = 2):
        """
        Enhanced U-Net architecture for PDE solution upscaling.
        
        Args:
            in_channels: Number of input channels (coarse solution + f)
        """
        super().__init__()
        
        # Encoder (moderate depth)
        self.enc1 = ConvBlock(in_channels, 64)    # 40×40
        self.enc2 = ConvBlock(64, 128)           # 20×20
        self.enc3 = ConvBlock(128, 256)          # 10×10
        
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
        self.dec3 = ConvBlock(512 + 256, 256)    # 10×10
        self.dec2 = ConvBlock(256 + 128, 128)    # 20×20
        self.dec1 = ConvBlock(128 + 64, 64)      # 40×40
        
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input channels
        coarse_solution = x[:, 0:1, :, :]
        features = x[:, 1:, :, :]  # Now only contains f
        
        # Encoder
        e1 = self.enc1(x)                    # 40×40
        e2 = self.enc2(self.pool(e1))        # 20×20
        e3 = self.enc3(self.pool(e2))        # 10×10
        
        # Bridge
        b = self.bridge(e3)                  # 10×10
        
        # Decoder with attention and skip connections
        e3_att = self.att3(e3, b)
        d3 = self.dec3(torch.cat([b, e3_att], dim=1))
        
        e2_att = self.att2(e2, self.up(d3))
        d2 = self.dec2(torch.cat([self.up(d3), e2_att], dim=1))
        
        e1_att = self.att1(e1, self.up(d2))
        d1 = self.dec1(torch.cat([self.up(d2), e1_att], dim=1))
        
        # Multi-scale refinement
        x = F.relu(self.out_bn1(self.out_conv1(d1)))
        x = F.relu(self.out_bn2(self.out_conv2(x)))
        x = self.final(x)
        
        # Residual connection from coarse solution
        return x + coarse_solution

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

class PDEDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: dict, device: str = 'cuda'):
        """
        Dataset class for PDE solutions.
        
        Args:
            data_dict: Dictionary containing the dataset
            device: Device to store tensors on
        """
        self.device = device
        
        # Convert numpy arrays to tensors and move to device
        self.u_coarse = torch.from_numpy(data_dict['u_coarse']).float().to(device)
        self.u_fine = torch.from_numpy(data_dict['u_fine']).float().to(device)
        self.f_fine = torch.from_numpy(data_dict['f_fine']).float().to(device)
        
        # Compute normalization statistics
        self.u_mean = self.u_fine.mean()
        self.u_std = self.u_fine.std()
        self.f_mean = self.f_fine.mean()
        self.f_std = self.f_fine.std()
        
        # Normalize the data
        self.u_fine_norm = (self.u_fine - self.u_mean) / self.u_std
        self.u_coarse_norm = (self.u_coarse - self.u_mean) / self.u_std
        self.f_fine_norm = (self.f_fine - self.f_mean) / self.f_std
        
        # Upsample coarse solution to fine grid
        self.u_coarse_upsampled = F.interpolate(
            self.u_coarse_norm.unsqueeze(1),
            size=(40, 40),
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
        
        # Target is the normalized fine solution
        y = self.u_fine_norm[idx].unsqueeze(0)
        
        return x, y
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize the model output back to original scale."""
        return x * self.u_std + self.u_mean

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