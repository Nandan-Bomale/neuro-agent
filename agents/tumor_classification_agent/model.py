"""
model.py
--------
Model architectures for the Tumor Classification Agent.

Stage 1 — TumorTypeEnsemble
    Three pretrained backbones (EfficientNet-B4, ResNet-50, DenseNet-121) whose
    softmax outputs are averaged at inference time.  During Phase 1 training all
    backbone weights are frozen; during Phase 2 the last 30 layers of each
    backbone are unfrozen for fine-tuning.

Stage 2 — GliomaGradeClassifier
    Single EfficientNet-B4 head.  Same frozen / unfrozen phase logic.

Usage
-----
    from agents.tumor_classification_agent.model import (
        TumorTypeEnsemble,
        GliomaGradeClassifier,
        build_type_ensemble,
        build_grade_classifier,
    )

    # Build models (weights loaded from torchvision hub if not cached)
    ensemble = build_type_ensemble(num_classes=4)
    grader   = build_grade_classifier(num_classes=3)

    # Phase 1 — train only the classification heads
    ensemble.set_phase(1)

    # Phase 2 — unfreeze last 30 layers of each backbone
    ensemble.set_phase(2)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    EfficientNet_B4_Weights,
    ResNet50_Weights,
    DenseNet121_Weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freeze_all(model: nn.Module) -> None:
    """Freeze every parameter in *model*."""
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_last_n(model: nn.Module, n: int) -> None:
    """Unfreeze the last *n* parameter tensors of *model* (by order)."""
    params = list(model.parameters())
    for p in params[-n:]:
        p.requires_grad = True


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Backbone factories
# ---------------------------------------------------------------------------

def _build_efficientnet_b4(num_classes: int, dropout: float = 0.4) -> nn.Module:
    """EfficientNet-B4 with a custom classification head."""
    net = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes),
    )
    return net


def _build_resnet50(num_classes: int, dropout: float = 0.4) -> nn.Module:
    """ResNet-50 with a custom classification head."""
    net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes),
    )
    return net


def _build_densenet121(num_classes: int, dropout: float = 0.4) -> nn.Module:
    """DenseNet-121 with a custom classification head."""
    net = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    in_features = net.classifier.in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes),
    )
    return net


# ---------------------------------------------------------------------------
# Stage 1: Tumor Type Ensemble
# ---------------------------------------------------------------------------

class TumorTypeEnsemble(nn.Module):
    """Ensemble of EfficientNet-B4 + ResNet-50 + DenseNet-121 for tumor typing.

    Inference: the softmax outputs of all three backbones are averaged and
    the argmax of the mean gives the predicted class.

    Training phases
    ---------------
    Phase 1 (frozen):   Only the custom classification heads are trainable.
    Phase 2 (unfrozen): The last 30 parameter tensors of each backbone are
                        additionally unfrozen for end-to-end fine-tuning.

    Args:
        num_classes: Number of output classes (4 for type: glioma / meningioma
                     / no_tumor / pituitary).
        dropout:     Dropout probability applied inside the head.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.4) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.efficientnet = _build_efficientnet_b4(num_classes, dropout)
        self.resnet        = _build_resnet50(num_classes, dropout)
        self.densenet      = _build_densenet121(num_classes, dropout)

        # Start in Phase 1 (heads only)
        self.set_phase(1)

    # ── Phase control ────────────────────────────────────────────────────────

    def set_phase(self, phase: int, n_unfreeze: int = 30) -> None:
        """Switch training phase.

        Args:
            phase:      1 = freeze backbones (train heads only).
                        2 = unfreeze last *n_unfreeze* parameter tensors per
                            backbone.
            n_unfreeze: Number of parameter tensors to unfreeze per backbone
                        in Phase 2.  Default: 30.
        """
        if phase == 1:
            _freeze_all(self.efficientnet)
            _freeze_all(self.resnet)
            _freeze_all(self.densenet)

            # Re-enable head parameters
            for head in [
                self.efficientnet.classifier,
                self.resnet.fc,
                self.densenet.classifier,
            ]:
                for p in head.parameters():
                    p.requires_grad = True

        elif phase == 2:
            # Keep Phase-1 setup then additionally unfreeze last n layers
            self.set_phase(1)
            _unfreeze_last_n(self.efficientnet, n_unfreeze)
            _unfreeze_last_n(self.resnet,        n_unfreeze)
            _unfreeze_last_n(self.densenet,       n_unfreeze)

        else:
            raise ValueError(f"phase must be 1 or 2, got {phase}")

        print(
            f"[TumorTypeEnsemble] Phase {phase} | "
            f"Trainable params: {_count_trainable(self):,}"
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Averaged softmax logit-equivalent forward pass.

        Returns raw logits of the averaged softmax (so that
        ``F.cross_entropy`` still works correctly).

        During inference call :meth:`predict_proba` instead to get
        calibrated probabilities.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Tensor of shape ``(B, num_classes)`` — averaged softmax scores
            (values in [0, 1] that sum to 1 across the class dimension).
        """
        p_eff  = F.softmax(self.efficientnet(x), dim=1)
        p_res  = F.softmax(self.resnet(x),         dim=1)
        p_den  = F.softmax(self.densenet(x),        dim=1)
        return (p_eff + p_res + p_den) / 3.0

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return averaged softmax probabilities (alias for eval forward).

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Probability tensor of shape ``(B, num_classes)``.
        """
        with torch.no_grad():
            return self.forward(x)

    def individual_logits(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw logits from each backbone individually.

        Useful for training with per-backbone losses or uncertainty analysis.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Tuple of (efficientnet_logits, resnet_logits, densenet_logits),
            each of shape ``(B, num_classes)``.
        """
        return (
            self.efficientnet(x),
            self.resnet(x),
            self.densenet(x),
        )


# ---------------------------------------------------------------------------
# Stage 2: Glioma Grade Classifier
# ---------------------------------------------------------------------------

class GliomaGradeClassifier(nn.Module):
    """EfficientNet-B4 single-head classifier for glioma grading.

    Only activated downstream of Stage 1 when tumor_type == "glioma".

    Training phases follow the same frozen / unfrozen scheme as the ensemble.

    Args:
        num_classes: Number of grade classes (3: grade_II / grade_III / grade_IV).
        dropout:     Dropout probability.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.4) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.net = _build_efficientnet_b4(num_classes, dropout)
        self.set_phase(1)

    # ── Phase control ────────────────────────────────────────────────────────

    def set_phase(self, phase: int, n_unfreeze: int = 30) -> None:
        """Switch training phase.

        Args:
            phase:      1 = freeze backbone (train head only).
                        2 = unfreeze last *n_unfreeze* parameter tensors.
            n_unfreeze: Number of parameter tensors to unfreeze in Phase 2.
        """
        if phase == 1:
            _freeze_all(self.net)
            for p in self.net.classifier.parameters():
                p.requires_grad = True

        elif phase == 2:
            self.set_phase(1)
            _unfreeze_last_n(self.net, n_unfreeze)

        else:
            raise ValueError(f"phase must be 1 or 2, got {phase}")

        print(
            f"[GliomaGradeClassifier] Phase {phase} | "
            f"Trainable params: {_count_trainable(self):,}"
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Probability tensor of shape ``(B, num_classes)``.
        """
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def build_type_ensemble(
    num_classes: int = 4,
    dropout: float = 0.4,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> TumorTypeEnsemble:
    """Instantiate and optionally load a TumorTypeEnsemble.

    Args:
        num_classes:     Number of output classes (default 4).
        dropout:         Dropout probability.
        checkpoint_path: If given, load saved state dict from this path.
        device:          Target device.  Auto-detected if None.

    Returns:
        TumorTypeEnsemble — ready for training or inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TumorTypeEnsemble(num_classes=num_classes, dropout=dropout).to(device)

    if checkpoint_path is not None:
        ckpt  = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        print(f"[build_type_ensemble] Loaded checkpoint: {checkpoint_path}")

    return model


def build_grade_classifier(
    num_classes: int = 3,
    dropout: float = 0.4,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> GliomaGradeClassifier:
    """Instantiate and optionally load a GliomaGradeClassifier.

    Args:
        num_classes:     Number of grade classes (default 3).
        dropout:         Dropout probability.
        checkpoint_path: If given, load saved state dict from this path.
        device:          Target device.  Auto-detected if None.

    Returns:
        GliomaGradeClassifier — ready for training or inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GliomaGradeClassifier(num_classes=num_classes, dropout=dropout).to(device)

    if checkpoint_path is not None:
        ckpt  = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        print(f"[build_grade_classifier] Loaded checkpoint: {checkpoint_path}")

    return model
