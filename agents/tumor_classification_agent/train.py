"""
train.py
--------
Training script for both stages of the Tumor Classification Agent.

Usage
-----
# Stage 1 — Tumor Type (4 classes)
python -m agents.tumor_classification_agent.train \
    --stage type \
    --data-path data/tumor_classification/type/ \
    --epochs 40 \
    --save-dir models/tumor_classifier/

# Stage 2 — Glioma Grade (3 classes)
python -m agents.tumor_classification_agent.train \
    --stage grade \
    --data-path data/tumor_classification/grade/ \
    --brats-path data/raw/BraTS2020_TrainingData/ \
    --epochs 40 \
    --save-dir models/tumor_classifier/

Training strategy
-----------------
Phase 1 (first 15 epochs): Frozen backbone — only heads are trained.
                            High LR (1e-3) with CosineAnnealing.
Phase 2 (remaining epochs): Unfreeze last 30 layers — lower LR (1e-4).
                            CosineAnnealing restarts.

Hardware presets
----------------
RTX 3050 4 GB : --batch-size 16 (type) or --batch-size 8 (grade)
DGX           : --batch-size 64

All checkpoints include val_accuracy, val_loss, epoch, and class_names so
inference.py can load them without needing to know the dataset layout.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from agents.tumor_classification_agent.dataset import (
    get_grade_dataloaders,
    get_type_dataloaders,
)
from agents.tumor_classification_agent.model import (
    GliomaGradeClassifier,
    TumorTypeEnsemble,
    build_grade_classifier,
    build_type_ensemble,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE1_EPOCHS  = 15      # epochs with frozen backbone
PHASE2_LR_SCALE = 0.1   # phase-2 LR = phase-1 LR × this


# ---------------------------------------------------------------------------
# Label-smoothing cross-entropy loss
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing for better calibration.

    Args:
        smoothing: Label smoothing factor (0.0 = standard cross-entropy).
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes  = logits.size(-1)
        log_probs  = nn.functional.log_softmax(logits, dim=-1)
        # One-hot hard targets
        nll_loss   = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        # Uniform smooth targets
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _accuracy(logits_or_probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for a batch."""
    preds = logits_or_probs.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device:    torch.device,
    scaler:    GradScaler,
    is_ensemble: bool,
) -> Tuple[float, float]:
    """Run one training epoch.

    Returns:
        (avg_loss, avg_accuracy) for this epoch.
    """
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            if is_ensemble:
                # Ensemble forward returns averaged probabilities; we need
                # individual logits for the loss to avoid log(prob) instability.
                logits_eff, logits_res, logits_den = model.individual_logits(images)
                loss = (
                    criterion(logits_eff, labels)
                    + criterion(logits_res, labels)
                    + criterion(logits_den, labels)
                ) / 3.0
                probs = model(images).detach()
            else:
                logits = model(images)
                loss   = criterion(logits, labels)
                probs  = torch.softmax(logits, dim=1).detach()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc  += _accuracy(probs, labels)
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def _val_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    is_ensemble: bool,
) -> Tuple[float, float]:
    """Run one validation epoch.

    Returns:
        (avg_loss, avg_accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            if is_ensemble:
                logits_eff, logits_res, logits_den = model.individual_logits(images)
                loss = (
                    criterion(logits_eff, labels)
                    + criterion(logits_res, labels)
                    + criterion(logits_den, labels)
                ) / 3.0
                probs = model(images)
            else:
                logits = model(images)
                loss   = criterion(logits, labels)
                probs  = torch.softmax(logits, dim=1)

        total_loss += loss.item()
        total_acc  += _accuracy(probs, labels)
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model:       nn.Module,
    optimizer:   optim.Optimizer,
    epoch:       int,
    val_acc:     float,
    val_loss:    float,
    class_names: List[str],
    save_path:   Path,
) -> None:
    """Save a full training checkpoint."""
    checkpoint = {
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "val_accuracy": round(val_acc, 6),
        "val_loss":     round(val_loss, 6),
        "class_names":  class_names,
    }
    torch.save(checkpoint, save_path)
    print(f"  ✓ Saved: {save_path.name}  (val_acc={val_acc:.4f})")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    stage:       str,
    data_path:   str,
    epochs:      int      = 40,
    save_dir:    str      = "models/tumor_classifier/",
    batch_size:  Optional[int] = None,
    lr:          float    = 1e-3,
    num_workers: int      = 4,
    seed:        int      = 42,
    smoothing:   float    = 0.1,
    brats_path:  Optional[str] = None,
) -> None:
    """Full training run for one stage.

    Args:
        stage:       "type" or "grade".
        data_path:   Path to dataset root (ImageFolder layout).
        epochs:      Total number of training epochs.
        save_dir:    Directory to save checkpoints.
        batch_size:  Mini-batch size.  Auto-selected if None.
        lr:          Initial learning rate for Phase 1.
        num_workers: DataLoader worker processes.
        seed:        Random seed.
        smoothing:   Label smoothing factor for LabelSmoothingCrossEntropy.
    """
    assert stage in ("type", "grade"), f"--stage must be 'type' or 'grade', got '{stage}'"

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train.py] Stage={stage} | Device={device} | Epochs={epochs}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if stage == "type":
        default_bs = 16
        bs = batch_size or default_bs
        # get_type_dataloaders returns (train, val, test, class_names)
        train_loader, val_loader, _test_loader, class_names = get_type_dataloaders(
            data_path, batch_size=bs, num_workers=num_workers, seed=seed
        )
        model       = build_type_ensemble(num_classes=len(class_names), device=device)
        is_ensemble = True
        prefix      = "type_ensemble"

    else:  # grade
        if brats_path is None:
            raise ValueError(
                "--brats-path is required for --stage grade. "
                "Example: --brats-path data/raw/BraTS2020_TrainingData/"
            )
        default_bs = 8
        bs = batch_size or default_bs
        # get_grade_dataloaders returns (train, val, class_names)
        train_loader, val_loader, class_names = get_grade_dataloaders(
            grade_root=data_path,
            brats_root=brats_path,
            batch_size=bs,
            num_workers=num_workers,
            seed=seed,
        )
        model       = build_grade_classifier(num_classes=len(class_names), device=device)
        is_ensemble = False
        prefix      = "grade_classifier"

    print(f"[train.py] Classes: {class_names}")

    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    scaler    = GradScaler()

    best_val_acc  = 0.0
    best_ckpt_path = save_dir_path / f"{prefix}_best.pth"

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    phase1_epochs = min(PHASE1_EPOCHS, epochs)
    model.set_phase(1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=lr * 1e-3)

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Frozen backbone ({phase1_epochs} epochs, LR={lr})")
    print(f"{'='*60}")

    for epoch in range(1, phase1_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, is_ensemble
        )
        vl_loss, vl_acc = _val_epoch(
            model, val_loader, criterion, device, is_ensemble
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:3d}/{phase1_epochs} | "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
            f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            _save_checkpoint(
                model, optimizer, epoch, vl_acc, vl_loss, class_names, best_ckpt_path
            )

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    if epochs > phase1_epochs:
        phase2_epochs = epochs - phase1_epochs
        phase2_lr     = lr * PHASE2_LR_SCALE

        model.set_phase(2)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=phase2_lr, weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=phase2_lr * 1e-3)

        print(f"\n{'='*60}")
        print(f"  PHASE 2 — Partial unfreeze ({phase2_epochs} epochs, LR={phase2_lr})")
        print(f"{'='*60}")

        for epoch in range(phase1_epochs + 1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = _train_epoch(
                model, train_loader, criterion, optimizer, device, scaler, is_ensemble
            )
            vl_loss, vl_acc = _val_epoch(
                model, val_loader, criterion, device, is_ensemble
            )
            scheduler.step()
            elapsed = time.time() - t0

            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
                f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} | "
                f"{elapsed:.1f}s"
            )

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                _save_checkpoint(
                    model, optimizer, epoch, vl_acc, vl_loss, class_names, best_ckpt_path
                )

            # Also save periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                periodic_path = save_dir_path / f"{prefix}_epoch{epoch:03d}.pth"
                _save_checkpoint(
                    model, optimizer, epoch, vl_acc, vl_loss, class_names, periodic_path
                )

    # ── Save class-name metadata ──────────────────────────────────────────────
    meta_path = save_dir_path / f"{prefix}_classes.json"
    meta_path.write_text(json.dumps({"class_names": class_names}))
    print(f"\n[train.py] Training complete. Best val_acc={best_val_acc:.4f}")
    print(f"[train.py] Best checkpoint: {best_ckpt_path}")
    print(f"[train.py] Class metadata : {meta_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Tumor Classification Agent (Stage 1: type | Stage 2: grade)"
    )
    p.add_argument(
        "--stage",
        required=True,
        choices=["type", "grade"],
        help="Training stage: 'type' for tumor type ensemble, 'grade' for glioma grader.",
    )
    p.add_argument(
        "--data-path",
        required=True,
        help=(
            "Stage 'type': path to data/tumor_classification/type/ "
            "(contains Training/ and Testing/ sub-folders). "
            "Stage 'grade': path to data/tumor_classification/grade/ "
            "(contains kaggle_3m/ sub-folder)."
        ),
    )
    p.add_argument(
        "--brats-path",
        default=None,
        help=(
            "(Grade stage only) Path to data/raw/BraTS2020_TrainingData/. "
            "Must contain MICCAI_BraTS2020_TrainingData/ with name_mapping.csv "
            "and per-patient NIfTI folders. Required when --stage grade."
        ),
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Total training epochs (Phase 1 = 15, Phase 2 = remaining). Default: 40.",
    )
    p.add_argument(
        "--save-dir",
        default="models/tumor_classifier/",
        help="Directory to save checkpoints. Default: models/tumor_classifier/",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size. Auto-selected (16 / 8) if not specified.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial Phase-1 learning rate. Default: 1e-3.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes. Default: 4.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor. Default: 0.1.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        stage       = args.stage,
        data_path   = args.data_path,
        epochs      = args.epochs,
        save_dir    = args.save_dir,
        batch_size  = args.batch_size,
        lr          = args.lr,
        num_workers = args.num_workers,
        seed        = args.seed,
        smoothing   = args.label_smoothing,
        brats_path  = args.brats_path,
    )
