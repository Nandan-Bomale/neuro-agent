"""
model.py
--------
3D CNN model architecture for Radiogenomics Agent.
Predicts IDH mutation and MGMT methylation status from 4-modality 3D MRI volumes.

Fixes applied:
  - Added Dropout(0.5) before final classifier to prevent overfitting on 585 samples
"""

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121


def build_radiogenomics_model(device: torch.device = None) -> nn.Module:
    """
    Builds a 3D DenseNet121 for predicting IDH and MGMT status.

    Architecture:
      - 3D DenseNet121 (MONAI implementation) with out_channels=1 (feature extractor)
      - Dropout(0.5) to prevent memorization on the small 585-sample dataset
      - Linear(1024 → 2) classification head
      - in_channels = 4 (FLAIR, T1, T1ce, T2)
      - spatial_dims = 3

    Output is raw logits. Use BCEWithLogitsLoss during training.
    """

    class RadiogenomicsNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Use DenseNet as backbone; set out_channels=1024 to get feature vector
            self.backbone = DenseNet121(
                spatial_dims=3,
                in_channels=4,
                out_channels=1024,
                pretrained=False,
            )
            # Strong dropout prevents memorizing the 585-sample training set
            self.head = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone(x)   # (B, 1024)
            return self.head(features)    # (B, 2)

    model = RadiogenomicsNet()

    if device is not None:
        model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[RadiogenomicsNet] Params: {total_params:,}")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_radiogenomics_model(device)
    dummy_input = torch.randn(2, 4, 128, 128, 128, device=device)
    print("Model initialized. Running forward pass with dummy input...")
    with torch.no_grad():
        out = model(dummy_input)
    print(f"Output shape: {out.shape} (Expected: [2, 2])")
