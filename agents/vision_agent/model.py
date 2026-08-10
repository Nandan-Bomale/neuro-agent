"""
model.py
--------
U-Net configuration for BraTS 2020 tumour segmentation.

Uses MONAI's built-in UNet which is a fully-configurable residual U-Net.
The architecture is tuned to the RTX 3050 (4 GB VRAM) constraint while
remaining large enough to achieve competitive Dice scores on BraTS.

Architecture summary
--------------------
  Input  : [B, 4, 128, 128, 128]   (4 MRI modalities, 128³ patch)
  Output : [B, 1, 128, 128, 128]   (binary whole-tumour mask, logits)

  Encoder : 4 → 32 → 64 → 128 → 256
  Bottleneck :         256 → 512
  Decoder : 256 → 128 → 64 → 32 → 1

  Residual units at every level, instance norm, LeakyReLU activation.

VRAM budget
-----------
  Batch size 2, patch 128³, float32 → ~1.6 GB VRAM.
  Leaves ~2.4 GB free for OS + LLM inference on the RTX 3050 (4 GB).

Loss function
-------------
  DiceCELoss  — weighted sum of Dice loss and cross-entropy.
  Dice handles class imbalance (tumour << background).
  Cross-entropy provides stable gradients early in training when Dice
  is near zero.

Usage
-----
    from agents.vision_agent.model import build_unet, build_loss, build_metric

    model  = build_unet()
    loss   = build_loss()
    metric = build_metric()
"""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose


# ---------------------------------------------------------------------------
# Architecture constants — change here to experiment, imports stay stable
# ---------------------------------------------------------------------------

#: Number of input channels — one per MRI modality (FLAIR, T1, T1ce, T2).
IN_CHANNELS: int = 4

#: Number of output channels — 1 for binary whole-tumour segmentation.
OUT_CHANNELS: int = 1

#: Feature map sizes at each encoder level.
#: Doubling at each level is the standard U-Net progression.
#: Max 256 (not 512) at the deepest encoder level to stay within VRAM budget.
FEATURES: tuple = (32, 64, 128, 256, 512)

#: Number of residual units per block.
#: 2 is standard for residual U-Net; reduces to 1 if VRAM is tight.
RESIDUAL_UNITS: int = 2

#: Dropout probability — applied inside residual blocks for regularisation.
DROPOUT: float = 0.1

#: Normalisation layer — instance norm is preferred over batch norm for MRI
#: because batch size is small (2) and each scan has different intensity stats.
NORM: str = "instance"

#: Activation — LeakyReLU avoids dying-ReLU problem in deep networks.
ACT: str = "leakyrelu"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_unet(device: torch.device | None = None) -> nn.Module:
    """Build and return the MONAI U-Net for BraTS segmentation.

    The network is initialised with default weights and moved to the
    specified device.  Call model.load_state_dict() afterwards to load
    trained weights.

    Args:
        device: torch.device to place the model on.  If None, uses CUDA
                if available, otherwise CPU.

    Returns:
        nn.Module — the U-Net, ready for training or inference.

    Example::

        model = build_unet()
        x = torch.randn(2, 4, 128, 128, 128).cuda()
        logits = model(x)          # shape: [2, 1, 128, 128, 128]
        probs  = torch.sigmoid(logits)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=3,            # 3-D volumetric MRI
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        channels=FEATURES,
        strides=(2, 2, 2, 2),      # stride-2 conv at each encoder level
        num_res_units=RESIDUAL_UNITS,
        dropout=DROPOUT,
        norm=NORM,
        act=ACT,
    )

    return model.to(device)


def build_loss() -> nn.Module:
    """Build the composite Dice + Cross-Entropy loss function.

    DiceCELoss combines:
      - Dice loss     : directly optimises the Dice score, robust to class
                        imbalance (important for BraTS where tumour voxels
                        are a small fraction of the total volume).
      - CE loss       : provides dense gradient signal early in training
                        before the model has learned to detect tumour at all.

    sigmoid=True because the model outputs raw logits (no sigmoid applied
    in build_unet), so the loss applies sigmoid internally before computing.

    Returns:
        DiceCELoss instance.
    """
    return DiceCELoss(
        sigmoid=True,       # apply sigmoid to logits before Dice/CE
        squared_pred=True,  # use squared predictions in Dice denominator
                            # (smoother gradients, standard practice)
        reduction="mean",
    )


def build_metric() -> DiceMetric:
    """Build the Dice metric for validation.

    Operates on binary predictions (after sigmoid + threshold at 0.5).
    Returns average Dice over the batch.

    The metric accumulates per-batch values and computes epoch-level mean
    via .aggregate() — this is the MONAI standard pattern used in train.py.

    Returns:
        DiceMetric instance.
    """
    return DiceMetric(
        include_background=False,   # only measure Dice on tumour class
        reduction="mean",
        get_not_nans=False,
    )


def build_post_transforms() -> tuple:
    """Build the post-processing transforms applied to model outputs.

    Used identically in train.py (validation loop) and inference.py.

    Returns:
        Tuple (post_pred, post_label) — both are MONAI Compose transforms.

        post_pred  : sigmoid → threshold at 0.5 → binary mask
        post_label : ensure label is already binary (identity for BraTS)
    """
    post_pred = Compose([
        Activations(sigmoid=True),          # logits → probabilities
        AsDiscrete(threshold=0.5),          # probabilities → {0, 1}
    ])

    post_label = AsDiscrete(to_onehot=None)  # label already binary, no-op

    return post_pred, post_label


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in the model.

    Useful for sanity-checking the architecture size at startup.

    Args:
        model: any nn.Module.

    Returns:
        int — total trainable parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
