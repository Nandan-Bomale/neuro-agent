"""
model.py
--------
Model architectures for the Tumor Classification Agent.

Stage 1 — TumorTypeEnsemble
    Three pretrained ImageNet backbones whose softmax outputs are averaged:
        • EfficientNet-B4
        • ResNet-50
        • DenseNet-121

    Each backbone gets a custom classification head:
        EfficientNet-B4 : AdaptiveAvgPool → Dropout(0.4) → Linear(1792→256)
                           → ReLU → Dropout(0.2) → Linear(256→4)
        ResNet-50        : AdaptiveAvgPool → Dropout(0.4) → Linear(2048→256)
                           → ReLU → Dropout(0.2) → Linear(256→4)
        DenseNet-121     : AdaptiveAvgPool → Dropout(0.4) → Linear(1024→256)
                           → ReLU → Dropout(0.2) → Linear(256→4)

    Training phases:
        Phase 1 — all backbone weights frozen; only heads trained (high LR).
        Phase 2 — last 30 parameter tensors of each backbone unfrozen (low LR).

    Key methods:
        forward(x)                 → averaged softmax probabilities (B, 4)
        individual_logits(x)       → raw logits per backbone, for per-backbone loss
        get_individual_models()    → [eff_b4, resnet50, densenet121] for separate optimisers
        set_phase(1 | 2)           → switch freeze / unfreeze mode

Stage 2 — TumorGradeClassifier
    Single EfficientNet-B4 with a custom grading head:
        GAP → Dropout(0.4) → Linear(1792→256) → ReLU → Linear(256→3)

    Same set_phase(1 | 2) interface as the ensemble.

Convenience constructors:
    build_type_ensemble(...)      → TumorTypeEnsemble
    build_grade_classifier(...)   → TumorGradeClassifier
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B4_Weights,
    ResNet50_Weights,
)


# ---------------------------------------------------------------------------
# Internal freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def _freeze_all(module: nn.Module) -> None:
    """Set requires_grad=False on every parameter of *module*."""
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze_all(module: nn.Module) -> None:
    """Set requires_grad=True on every parameter of *module*."""
    for p in module.parameters():
        p.requires_grad = True


def _unfreeze_last_n_layers(module: nn.Module, n: int) -> None:
    """Unfreeze the last *n* parameter tensors (by iteration order) in *module*.

    The iteration order of ``module.parameters()`` follows the order in which
    sub-modules are registered, which corresponds to the forward-pass order for
    all standard torchvision backbones.  Unfreezing the *tail* of this list
    therefore unfreezes the deepest layers — exactly what we want for fine-tuning.

    Args:
        module: Any nn.Module whose deepest layers should be unfrozen.
        n:      Number of parameter tensors to unfreeze.  A value of 30 covers
                roughly the last 2-3 residual blocks / dense blocks plus BN.
    """
    params = list(module.parameters())
    for p in params[-n:]:
        p.requires_grad = True


def _count_trainable(module: nn.Module) -> int:
    """Return the number of trainable scalar parameters in *module*."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Custom classification head factory
# ---------------------------------------------------------------------------

def _make_head(in_features: int, num_classes: int, dropout: float = 0.4) -> nn.Sequential:
    """Shared head design used by all three Stage-1 backbones.

    Architecture:
        Dropout(dropout) → Linear(in_features→256) → ReLU → Dropout(dropout/2)
        → Linear(256→num_classes)

    The AdaptiveAvgPool is handled by each backbone's own pooling layer
    before the head receives a flat (B, in_features) tensor.

    Args:
        in_features: Output width from the backbone's pooling/feature layer.
        num_classes: Number of output classes.
        dropout:     Primary dropout rate; secondary = dropout / 2.

    Returns:
        nn.Sequential classification head.
    """
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout / 2),
        nn.Linear(256, num_classes),
    )


# ---------------------------------------------------------------------------
# Individual backbone constructors (return full model with replaced head)
# ---------------------------------------------------------------------------

def _build_efficientnet_b4(num_classes: int, dropout: float = 0.4) -> nn.Module:
    """EfficientNet-B4 with ImageNet weights and a custom classification head.

    The standard EfficientNet classifier is replaced with our shared head.
    The backbone already ends with AdaptiveAvgPool → Flatten internally,
    so the head receives a (B, 1792) tensor.

    Args:
        num_classes: Output class count.
        dropout:     Primary dropout rate in the head.

    Returns:
        nn.Module — full EfficientNet-B4 with custom head.
    """
    net = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = net.classifier[1].in_features   # 1792
    net.classifier = _make_head(in_features, num_classes, dropout)
    return net


def _build_resnet50(num_classes: int, dropout: float = 0.4) -> nn.Module:
    """ResNet-50 with ImageNet weights and a custom classification head.

    The standard ``fc`` layer is replaced with our shared head.
    AdaptiveAvgPool2d is already applied inside ResNet before ``fc``,
    so the head receives a (B, 2048) tensor.

    Args:
        num_classes: Output class count.
        dropout:     Primary dropout rate in the head.

    Returns:
        nn.Module — full ResNet-50 with custom head.
    """
    net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = net.fc.in_features               # 2048
    net.fc = _make_head(in_features, num_classes, dropout)
    return net


def _build_densenet121(num_classes: int, dropout: float = 0.4) -> nn.Module:
    """DenseNet-121 with ImageNet weights and a custom classification head.

    The standard ``classifier`` linear layer is replaced with our shared head.
    DenseNet applies AdaptiveAvgPool + Flatten in its forward pass before the
    classifier, so the head receives a (B, 1024) tensor.

    Args:
        num_classes: Output class count.
        dropout:     Primary dropout rate in the head.

    Returns:
        nn.Module — full DenseNet-121 with custom head.
    """
    net = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    in_features = net.classifier.in_features       # 1024
    net.classifier = _make_head(in_features, num_classes, dropout)
    return net


# ===========================================================================
# Stage 1 — TumorTypeEnsemble
# ===========================================================================

class TumorTypeEnsemble(nn.Module):
    """Ensemble of EfficientNet-B4 + ResNet-50 + DenseNet-121 for tumor typing.

    At inference the softmax outputs of all three backbones are averaged
    and argmax gives the predicted class.

    At training time ``individual_logits()`` is used to compute per-backbone
    cross-entropy losses (which sum/average to a single scalar) so gradients
    flow correctly through each backbone.

    Training phases
    ---------------
    Phase 1 — All backbone weights frozen.  Only the three custom heads are
               updated.  Use a high LR (e.g., 1e-3).
    Phase 2 — Backbone weights frozen + last 30 parameter tensors of each
               backbone unfrozen for fine-tuning.  Use a low LR (e.g., 1e-4).

    Args:
        num_classes: Number of type classes (4: glioma/meningioma/notumor/pituitary).
        dropout:     Primary dropout rate inside each head.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.4) -> None:
        super().__init__()
        self.num_classes = num_classes

        # ── Three pretrained backbones ────────────────────────────────────────
        self.efficientnet_b4 = _build_efficientnet_b4(num_classes, dropout)
        self.resnet50        = _build_resnet50(num_classes, dropout)
        self.densenet121     = _build_densenet121(num_classes, dropout)

        # Start in Phase 1 (frozen backbones, trainable heads only)
        self.set_phase(1)

    # ── Phase control ────────────────────────────────────────────────────────

    def set_phase(self, phase: int, n_unfreeze: int = 30) -> None:
        """Switch the training phase.

        Args:
            phase:      1 → freeze all backbones, train heads only.
                        2 → additionally unfreeze the last *n_unfreeze*
                            parameter tensors of each backbone.
            n_unfreeze: Number of parameter tensors to unfreeze per backbone
                        in Phase 2.  Default: 30.

        Raises:
            ValueError: if phase is not 1 or 2.
        """
        if phase not in (1, 2):
            raise ValueError(f"phase must be 1 or 2, got {phase}")

        # --- freeze everything first ---
        for backbone, head_attr in [
            (self.efficientnet_b4, "classifier"),
            (self.resnet50,        "fc"),
            (self.densenet121,     "classifier"),
        ]:
            _freeze_all(backbone)
            # Re-enable the custom head
            _unfreeze_all(getattr(backbone, head_attr))

        if phase == 2:
            # Additionally unfreeze the deepest n_unfreeze param tensors
            # of each backbone (not just the head — includes late BN/conv layers)
            for backbone in (self.efficientnet_b4, self.resnet50, self.densenet121):
                _unfreeze_last_n_layers(backbone, n_unfreeze)

        total_trainable = _count_trainable(self)
        eff_t  = _count_trainable(self.efficientnet_b4)
        res_t  = _count_trainable(self.resnet50)
        den_t  = _count_trainable(self.densenet121)
        print(
            f"[TumorTypeEnsemble] Phase {phase} activated | "
            f"Trainable: total={total_trainable:,} "
            f"(eff={eff_t:,} res={res_t:,} den={den_t:,})"
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Averaged softmax probabilities across all three backbones.

        This is the *inference* forward.  For training use ``individual_logits``
        to compute proper cross-entropy on each backbone's raw logits.

        Args:
            x: Float tensor of shape ``(B, 3, H, W)``.

        Returns:
            Probability tensor of shape ``(B, num_classes)`` summing to 1
            along dim=1.
        """
        p_eff = F.softmax(self.efficientnet_b4(x), dim=1)
        p_res = F.softmax(self.resnet50(x),        dim=1)
        p_den = F.softmax(self.densenet121(x),     dim=1)
        return (p_eff + p_res + p_den) / 3.0

    def individual_logits(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw logits from each backbone separately.

        Used during training to compute per-backbone cross-entropy losses:

            loss = (CE(logits_eff, y) + CE(logits_res, y) + CE(logits_den, y)) / 3

        This ensures gradients flow through all three backbones independently.

        Args:
            x: Float tensor of shape ``(B, 3, H, W)``.

        Returns:
            Tuple of three logit tensors, each of shape ``(B, num_classes)``:
            (efficientnet_b4_logits, resnet50_logits, densenet121_logits).
        """
        return (
            self.efficientnet_b4(x),
            self.resnet50(x),
            self.densenet121(x),
        )

    def get_individual_models(self) -> List[nn.Module]:
        """Return a list of the three backbone models.

        Useful for constructing per-backbone optimisers during training or for
        independent evaluation of each model's contribution to the ensemble.

        Returns:
            [efficientnet_b4, resnet50, densenet121] — in that fixed order.
        """
        return [self.efficientnet_b4, self.resnet50, self.densenet121]

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Averaged softmax probabilities with no_grad wrapper.

        Convenience method for inference — identical to ``forward`` but
        wrapped in ``torch.no_grad``.

        Args:
            x: Float tensor of shape ``(B, 3, H, W)``.

        Returns:
            Probability tensor of shape ``(B, num_classes)``.
        """
        with torch.no_grad():
            return self.forward(x)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        n_train  = _count_trainable(self)
        return (
            f"TumorTypeEnsemble("
            f"num_classes={self.num_classes}, "
            f"total_params={n_params:,}, "
            f"trainable_params={n_train:,})"
        )


# ===========================================================================
# Stage 2 — TumorGradeClassifier
# ===========================================================================

class TumorGradeClassifier(nn.Module):
    """EfficientNet-B4 grading head for glioma grade classification.

    Only activated downstream of Stage 1 when tumor_type == "glioma".

    Head architecture (replaces the default EfficientNet classifier):
        AdaptiveAvgPool (done inside EfficientNet features)
        → Flatten
        → Dropout(0.4)
        → Linear(1792 → 256)
        → ReLU
        → Linear(256 → num_classes)

    Training phases:
        Phase 1 — backbone frozen, head trainable.
        Phase 2 — additionally unfreeze last *n_unfreeze* param tensors.

    Args:
        num_classes: Grade classes (3: grade_II / grade_III / grade_IV).
        dropout:     Primary dropout rate in the head.
    """

    # EfficientNet-B4 feature width before the classifier
    _EFF_B4_FEATURES = 1792

    def __init__(self, num_classes: int = 3, dropout: float = 0.4) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained backbone
        backbone = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        # Keep feature extractor (includes AdaptiveAvgPool + Flatten in forward)
        self.features = backbone.features   # Conv + BN + MBConv blocks
        self.avgpool  = backbone.avgpool    # AdaptiveAvgPool2d(1, 1)

        # Custom grading head: GAP → Dropout → Linear(256) → ReLU → Linear(num_classes)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self._EFF_B4_FEATURES, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        # Start in Phase 1
        self.set_phase(1)

    # ── Phase control ────────────────────────────────────────────────────────

    def set_phase(self, phase: int, n_unfreeze: int = 30) -> None:
        """Switch the training phase.

        Args:
            phase:      1 → freeze feature extractor, train head only.
                        2 → additionally unfreeze last *n_unfreeze* param
                            tensors of the feature extractor.
            n_unfreeze: Number of param tensors to unfreeze in Phase 2.

        Raises:
            ValueError: if phase is not 1 or 2.
        """
        if phase not in (1, 2):
            raise ValueError(f"phase must be 1 or 2, got {phase}")

        # Freeze everything, then enable head
        _freeze_all(self.features)
        _freeze_all(self.avgpool)
        _unfreeze_all(self.head)

        if phase == 2:
            _unfreeze_last_n_layers(self.features, n_unfreeze)

        total_trainable = _count_trainable(self)
        print(
            f"[TumorGradeClassifier] Phase {phase} activated | "
            f"Trainable params: {total_trainable:,}"
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits.

        Pipeline: features → avgpool → flatten → head.

        Args:
            x: Float tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        feats  = self.features(x)          # (B, 1792, h, w)
        pooled = self.avgpool(feats)        # (B, 1792, 1, 1)
        logits = self.head(pooled)          # (B, num_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities with no_grad wrapper.

        Args:
            x: Float tensor of shape ``(B, 3, H, W)``.

        Returns:
            Probability tensor of shape ``(B, num_classes)``.
        """
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        n_train  = _count_trainable(self)
        return (
            f"TumorGradeClassifier("
            f"num_classes={self.num_classes}, "
            f"total_params={n_params:,}, "
            f"trainable_params={n_train:,})"
        )


# ===========================================================================
# Convenience constructors
# ===========================================================================

def build_type_ensemble(
    num_classes:     int                    = 4,
    dropout:         float                  = 0.4,
    checkpoint_path: Optional[str]          = None,
    device:          Optional[torch.device] = None,
) -> TumorTypeEnsemble:
    """Build and optionally load a TumorTypeEnsemble.

    Args:
        num_classes:     Number of type classes (default 4).
        dropout:         Head dropout rate.
        checkpoint_path: If provided, load state dict from this .pth file.
                         Checkpoint format: ``{"model_state": ..., ...}``
                         or a bare state dict.
        device:          Target device.  Auto-detected (CUDA if available).

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
        epoch = ckpt.get("epoch", "?")
        acc   = ckpt.get("val_accuracy", "?")
        print(
            f"[build_type_ensemble] Loaded: {checkpoint_path} "
            f"(epoch={epoch}, val_acc={acc})"
        )

    return model


def build_grade_classifier(
    num_classes:     int                    = 3,
    dropout:         float                  = 0.4,
    checkpoint_path: Optional[str]          = None,
    device:          Optional[torch.device] = None,
) -> TumorGradeClassifier:
    """Build and optionally load a TumorGradeClassifier.

    Args:
        num_classes:     Number of grade classes (default 3).
        dropout:         Head dropout rate.
        checkpoint_path: If provided, load state dict from this .pth file.
        device:          Target device.  Auto-detected (CUDA if available).

    Returns:
        TumorGradeClassifier — ready for training or inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TumorGradeClassifier(num_classes=num_classes, dropout=dropout).to(device)

    if checkpoint_path is not None:
        ckpt  = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        epoch = ckpt.get("epoch", "?")
        acc   = ckpt.get("val_accuracy", "?")
        print(
            f"[build_grade_classifier] Loaded: {checkpoint_path} "
            f"(epoch={epoch}, val_acc={acc})"
        )

    return model
