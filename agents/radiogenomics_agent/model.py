"""
model.py
--------
3D CNN model architecture for Radiogenomics Agent.
Predicts IDH mutation and MGMT methylation status from 4-modality 3D MRI volumes.
"""

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

def build_radiogenomics_model(device: torch.device = None) -> nn.Module:
    """
    Builds a 3D DenseNet121 for predicting IDH and MGMT status.
    
    Architecture:
      - 3D DenseNet121 (MONAI implementation)
      - in_channels = 4 (FLAIR, T1, T1ce, T2)
      - out_channels = 2 (IDH, MGMT)
      - spatial_dims = 3
    
    Output is raw logits. Use BCEWithLogitsLoss during training.
    """
    model = DenseNet121(
        spatial_dims=3,
        in_channels=4,
        out_channels=2,
        pretrained=False  # No direct 3D 4-channel pretrained weights by default in MONAI
    )
    
    if device is not None:
        model = model.to(device)
        
    return model

if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_radiogenomics_model(device)
    dummy_input = torch.randn(2, 4, 128, 128, 128, device=device)
    print("Model initialized. Running forward pass with dummy input...")
    with torch.no_grad():
        out = model(dummy_input)
    print(f"Output shape: {out.shape} (Expected: [2, 2])")
