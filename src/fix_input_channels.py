#!/usr/bin/env python

"""
This script fixes the input channel mismatch in resolution_comparison.py to match
the expected 2 channels for the UNet model.
"""

import re

# Read the original file
with open('resolution_comparison.py', 'r') as f:
    content = f.read()

# Fix the input channel preparation in the upscale_subdomain function
pattern = r'# Ensure proper dimensions for upsampling.*?# Combine inputs \(only coarse solution and f\)\s+inputs = torch\.cat\(\[\s+u_coarse_upsampled\.squeeze\(0\),\s+f_fine_norm\s+\], dim=0\)'

replacement = '''# Ensure proper dimensions for upsampling
    # Reshape if needed to ensure we have [batch, channel, height, width]
    if len(u_coarse_norm.shape) == 2:
        u_coarse_norm = u_coarse_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif len(u_coarse_norm.shape) == 3:
        u_coarse_norm = u_coarse_norm.unsqueeze(0)  # [1, C, H, W]
    
    # Upsample coarse solution
    u_coarse_upsampled = F.interpolate(
        u_coarse_norm,
        size=(40, 40),
        mode='bilinear',
        align_corners=True
    )
    
    # Ensure proper dimensions for f_fine_norm
    if len(f_fine_norm.shape) == 2:
        f_fine_norm = f_fine_norm.unsqueeze(0)  # [1, H, W]
    
    # Combine inputs (only coarse solution and f) - ensure exactly 2 channels
    # First channel: upsampled coarse solution, Second channel: fine forcing
    inputs = torch.cat([
        u_coarse_upsampled.squeeze(0).squeeze(0).unsqueeze(0),  # [1, H, W]
        f_fine_norm.squeeze(0).unsqueeze(0)  # [1, H, W]
    ], dim=0)  # Result: [2, H, W]'''

# Apply the replacement using re.DOTALL to match across multiple lines
modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write the modified content back to the file
with open('resolution_comparison.py', 'w') as f:
    f.write(modified_content)

print("Successfully fixed input channel mismatch in resolution_comparison.py")
