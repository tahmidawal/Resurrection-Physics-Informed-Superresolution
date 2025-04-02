import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryAwareLoss(nn.Module):
    def __init__(self, boundary_weight=2.0, subdomain_size=40, boundary_width=3):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.subdomain_size = subdomain_size
        self.boundary_width = boundary_width

    def forward(self, outputs, targets):
        # Base MSE loss
        mse_loss = F.mse_loss(outputs, targets, reduction=none)

        # Create weight map for boundaries
        batch_size, channels, height, width = outputs.shape
        weight_map = torch.ones_like(outputs)

        # Apply higher weights to boundary regions
        if self.boundary_width > 0:
            # Apply boundary weights to all edges
            weight_map[:, :, :self.boundary_width, :] *= self.boundary_weight  # Top
            weight_map[:, :, -self.boundary_width:, :] *= self.boundary_weight  # Bottom
            weight_map[:, :, :, :self.boundary_width] *= self.boundary_weight  # Left
            weight_map[:, :, :, -self.boundary_width:] *= self.boundary_weight  # Right

        # Apply weights to the loss
        weighted_loss = mse_loss * weight_map

        # Return mean loss
        return weighted_loss.mean()
