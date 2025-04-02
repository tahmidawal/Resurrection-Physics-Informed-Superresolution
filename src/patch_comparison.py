#!/usr/bin/env python

"""
This script patches the resolution_comparison.py file to handle both list and numpy array inputs
for the overlapping subdomain implementation.
"""

import re

# Read the original file
with open('resolution_comparison.py', 'r') as f:
    content = f.read()

# Find and replace the upscale_subdomain function definition
pattern = r'def upscale_subdomain\(model: torch\.nn\.Module, u_coarse: np\.ndarray,\s+f_fine: np\.ndarray, theta_fine: np\.ndarray,\s+global_norm, device: str\) -> np\.ndarray:\s+"""[\s\S]+?# Convert inputs to tensors\s+u_coarse = torch\.from_numpy\(u_coarse\)\.float\(\)\.to\(device\)\s+f_fine = torch\.from_numpy\(f_fine\)\.float\(\)\.to\(device\)'

replacement = '''def upscale_subdomain(model: torch.nn.Module, u_coarse, 
                      f_fine, theta_fine: np.ndarray,
                      global_norm, device: str) -> np.ndarray:
    """
    Apply the ML model to upscale a single subdomain using global normalization.
    """
    # Convert inputs to tensors - handle both list and numpy array inputs
    if isinstance(u_coarse, list):
        u_coarse = np.array(u_coarse)
    if isinstance(f_fine, list):
        f_fine = np.array(f_fine)
        
    u_coarse = torch.from_numpy(u_coarse).float().to(device)
    f_fine = torch.from_numpy(f_fine).float().to(device)'''

# Apply the replacement
modified_content = re.sub(pattern, replacement, content)

# Write the modified content back to the file
with open('resolution_comparison.py', 'w') as f:
    f.write(modified_content)

print("Successfully patched resolution_comparison.py to handle list inputs for subdomains.")
