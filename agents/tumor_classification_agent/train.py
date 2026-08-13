"""
train.py
--------
Two-phase training script for the Tumor Classification Agent.

Stage 1 — Tumor Type (4 classes: glioma / meningioma / notumor / pituitary)
    Model  : TumorTypeEnsemble (EfficientNet-B4 + ResNet-50 + DenseNet-121)
    Dataset: data/tumor_classification/type/  (Training/ + Testing/ ImageFolder)
    Target : 98–99 % accuracy

Stage 2 — Glioma Grade (3 classes: grade_II / grade_III / grade_IV)
    Model  : TumorGradeClassifier (EfficientNet-B4)
    Dataset: data/tumor_classification/grade/kaggle_3m/  (Grade II TIF slices)
             + data/raw/BraTS2020_TrainingData/            (Grade III/IV NIfTI)
    Target : 90 %+ accuracy

Training strategy
-----------------
Phase 1 (first PHASE1_EPOCHS epochs):
    Backbone frozen, only heads trained.
    Optimiser : AdamW, LR = args.lr, weight_decay = 1e-4
    Scheduler : CosineAnnealingLR

Phase 2 (remaining epochs):
    Last 30 parameter tensors of each backbone unfrozen.
    Optimiser : AdamW, LR = args.lr × 0.1 (lower for fine-tuning)
    Scheduler : CosineAnnealingLR (restarts)

Mixed precision (torch.cuda.amp) is used throughout for RTX VRAM efficiency.
Gradient clipping (max_norm=1.0) prevents gradient explosions in Phase 2.

Ensemble training loss
----------------------
For Stage 1, the loss is the average of three per-backbone cross-entropy losses:
    L = (CE(eff_logits, y) + CE(res_logits, y) + CE(den_logits, y)) / 3
This ensures all three backbones receive independent gradients.

CLI
---
# Stage 1
python -m agents.tumor_classification_agent.train \\
    --stage type \\
    --data-path data/tumor_classification/type/ \\
    --epochs 40 --batch-size 16 \\
    --save-dir models/tumor_classifier/

# Stage 2
python -m agents.tumor_classification_agent.train \\
    --stage grade \\
    --data-path data/tumor_classification/grade/ \\
    --brats-path data/raw/BraTS2020_TrainingData/ \\
    --epochs 40 --batch-size 8 \\
    --save-dir models/tumor_classifier/

Hardware presets
----------------
RTX 3050 4 GB : --batch-size 16 (type) | --batch-size 8 (grade)
DGX A100      : --batch-size 64  (both stages)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from agents.tumor_classification_agent.dataset import (
    get_grade_dataloaders,
    get_type_dataloaders,
)
from agents.tumor_classification_agent.model import (
    TumorGradeClassifier,
    TumorTypeEnsemble,
    build_grade_classifier,
    build_type_ensemble,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE1_EPOCHS   = 15      # epochs with frozen backbone
PHASE2_LR_SCALE = 0.1    # phase-2 LR  =  phase-1 LR  ×  this factor
N_UNFREEZE      = 30      # param tensors to unfreeze per backbone in Phase 2


# ---------------------------------------------------------------------------
# Label-smoothing cross-entropy
# ---------------------------------------------------------------------------

class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing for better-calibrated outputs.

    Args:
        smoothing: Smoothing factor ε.  0 = standard cross-entropy.
        reduction: 'mean' (default) or 'sum'.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.smoothing  = smoothing
        self.reduction  = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Hard-target NLL
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        # Uniform smooth term
        smooth = -log_probs.mean(dim=-1)

        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean() if self.reduction == "mean" else loss.sum()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _top1_accuracy(probs_or_logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy for one batch."""
    preds = probs_or_logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _class_accuracy(
    probs_or_logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[int, float]:
    """Per-class accuracy for one batch."""
    preds = probs_or_logits.argmax(dim=1)
    acc: Dict[int, float] = {}
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() == 0:
            continue
        acc[c] = (preds[mask] == targets[mask]).float().mean().item()
    return acc


# ---------------------------------------------------------------------------
# One-epoch training / validation helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    optimizer:   optim.Optimizer,
    device:      torch.device,
    scaler:      GradScaler,
    is_ensemble: bool,
) -> Tuple[float, float]:
    """Run one full training epoch.

    For the ensemble, individual_logits() is used so cross-entropy flows
    through each backbone independently.  The averaged softmax is used only
    to compute the accuracy metric.

    Returns:
        (mean_loss, mean_accuracy) over all batches.
    """
    model.train()
    total_loss = total_acc = 0.0
    n_batches  = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            if is_ensemble:
                assert isinstance(model, TumorTypeEnsemble)
                eff_l, res_l, den_l = model.individual_logits(images)
                loss = (
                    criterion(eff_l, labels)
                    + criterion(res_l, labels)
                    + criterion(den_l, labels)
                ) / 3.0
                # Use averaged softmax for accuracy
                with torch.no_grad():
                    probs = model(images)
            else:
                logits = model(images)
                loss   = criterion(logits, labels)
                probs  = F.softmax(logits.detach(), dim=1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc  += _top1_accuracy(probs, labels)
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def _val_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    is_ensemble: bool,
    num_classes: int,
) -> Tuple[float, float, Dict[int, float]]:
    """Run one full validation epoch.

    Returns:
        (mean_loss, mean_accuracy, per_class_accuracy_dict).
    """
    model.eval()
    total_loss  = total_acc = 0.0
    n_batches   = 0
    class_accs: List[Dict[int, float]] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            if is_ensemble:
                assert isinstance(model, TumorTypeEnsemble)
                eff_l, res_l, den_l = model.individual_logits(images)
                loss  = (
                    criterion(eff_l, labels)
                    + criterion(res_l, labels)
                    + criterion(den_l, labels)
                ) / 3.0
                probs = model(images)
            else:
                logits = model(images)
                loss   = criterion(logits, labels)
                probs  = F.softmax(logits, dim=1)

        total_loss += loss.item()
        total_acc  += _top1_accuracy(probs, labels)
        class_accs.append(_class_accuracy(probs, labels, num_classes))
        n_batches  += 1

    # Merge per-class dicts
    merged: Dict[int, List[float]] = {}
    for d in class_accs:
        for c, a in d.items():
            merged.setdefault(c, []).append(a)
    per_class = {c: sum(v) / len(v) for c, v in merged.items()}

    return total_loss / n_batches, total_acc / n_batches, per_class


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
    path:        Path,
    extra:       Optional[dict] = None,
) -> None:
    """Serialize a full training checkpoint."""
    ckpt = {
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "val_accuracy": round(val_acc, 6),
        "val_loss":     round(val_loss, 6),
        "class_names":  class_names,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
    print(f"  ✓ checkpoint saved → {path.name}  (val_acc={val_acc:.4f})")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    stage:       str,
    data_path:   str,
    epochs:      int            = 40,
    save_dir:    str            = "models/tumor_classifier/",
    batch_size:  Optional[int]  = None,
    lr:          float          = 1e-3,
    num_workers: int            = 4,
    seed:        int            = 42,
    smoothing:   float          = 0.1,
    brats_path:  Optional[str]  = None,
) -> None:
    """Full two-phase training run for one stage.

    Args:
        stage:       "type" (Stage 1) or "grade" (Stage 2).
        data_path:   For type: path to type/ root (Training/ + Testing/).
                     For grade: path to grade/ root (kaggle_3m/).
        epochs:      Total epochs (Phase 1 = PHASE1_EPOCHS, Phase 2 = rest).
        save_dir:    Directory where checkpoints are saved.
        batch_size:  Mini-batch size; auto-selected if None (16 type, 8 grade).
        lr:          Phase-1 initial learning rate.
        num_workers: DataLoader worker count.
        seed:        Random seed.
        smoothing:   Label-smoothing factor.
        brats_path:  (Grade stage only) path to BraTS data root.
    """
    assert stage in ("type", "grade"), f"--stage must be 'type' or 'grade', got '{stage}'"

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*65}")
    print(f"  Tumor Classification — Stage: {stage.upper()} | Device: {device}")
    print(f"  Epochs: {epochs} | LR: {lr} | Seed: {seed}")
    print(f"{'='*65}\n")

    # ── Build DataLoaders ────────────────────────────────────────────────────
    if stage == "type":
        bs = batch_size or 16
        train_loader, val_loader, _test_loader, class_names = get_type_dataloaders(
            data_root=data_path,
            batch_size=bs,
            num_workers=num_workers,
            seed=seed,
        )
        model       = build_type_ensemble(num_classes=len(class_names), device=device)
        is_ensemble = True
        prefix      = "type_ensemble"

    else:  # grade
        if brats_path is None:
            raise ValueError(
                "--brats-path is required for --stage grade.\n"
                "Example: --brats-path data/raw/BraTS2020_TrainingData/"
            )
        bs = batch_size or 8
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

    num_classes = len(class_names)
    print(f"[train] classes ({num_classes}): {class_names}")
    print(f"[train] batch_size={bs} | train_batches={len(train_loader)} | val_batches={len(val_loader)}\n")

    criterion = LabelSmoothingCE(smoothing=smoothing)
    scaler    = GradScaler()
    best_val_acc   = 0.0
    best_ckpt_path = save_path / f"{prefix}_best.pth"

    # ── PHASE 1 — frozen backbone ────────────────────────────────────────────
    phase1_eps = min(PHASE1_EPOCHS, epochs)
    model.set_phase(1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_eps, eta_min=lr * 1e-3)

    print(f"\n{'─'*65}")
    print(f"  PHASE 1  |  Frozen backbone  |  {phase1_eps} epochs  |  LR={lr:.0e}")
    print(f"{'─'*65}")

    for epoch in range(1, phase1_eps + 1):
        t0 = time.time()
        tr_loss, tr_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, is_ensemble
        )
        vl_loss, vl_acc, vl_per_cls = _val_epoch(
            model, val_loader, criterion, device, is_ensemble, num_classes
        )
        scheduler.step()
        elapsed = time.time() - t0

        per_cls_str = "  ".join(
            f"{class_names[c]}={vl_per_cls[c]:.3f}"
            for c in sorted(vl_per_cls)
        )
        print(
            f"  Ep {epoch:3d}/{phase1_eps} │ "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} │ "
            f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} │ "
            f"{elapsed:.1f}s"
        )
        print(f"           per-class: {per_cls_str}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            _save_checkpoint(
                model, optimizer, epoch,
                vl_acc, vl_loss, class_names,
                best_ckpt_path,
                extra={"phase": 1},
            )

    # ── PHASE 2 — partial unfreeze ───────────────────────────────────────────
    if epochs > phase1_eps:
        phase2_eps = epochs - phase1_eps
        phase2_lr  = lr * PHASE2_LR_SCALE

        model.set_phase(2, n_unfreeze=N_UNFREEZE)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=phase2_lr, weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=phase2_eps, eta_min=phase2_lr * 1e-3)

        print(f"\n{'─'*65}")
        print(f"  PHASE 2  |  Last {N_UNFREEZE} layers unfrozen  |  {phase2_eps} epochs  |  LR={phase2_lr:.0e}")
        print(f"{'─'*65}")

        for epoch in range(phase1_eps + 1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = _train_epoch(
                model, train_loader, criterion, optimizer, device, scaler, is_ensemble
            )
            vl_loss, vl_acc, vl_per_cls = _val_epoch(
                model, val_loader, criterion, device, is_ensemble, num_classes
            )
            scheduler.step()
            elapsed = time.time() - t0

            per_cls_str = "  ".join(
                f"{class_names[c]}={vl_per_cls[c]:.3f}"
                for c in sorted(vl_per_cls)
            )
            print(
                f"  Ep {epoch:3d}/{epochs} │ "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} │ "
                f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} │ "
                f"{elapsed:.1f}s"
            )
            print(f"           per-class: {per_cls_str}")

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                _save_checkpoint(
                    model, optimizer, epoch,
                    vl_acc, vl_loss, class_names,
                    best_ckpt_path,
                    extra={"phase": 2},
                )

            # Periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                periodic = save_path / f"{prefix}_ep{epoch:03d}.pth"
                _save_checkpoint(
                    model, optimizer, epoch,
                    vl_acc, vl_loss, class_names, periodic,
                )

    # ── Save class-name metadata sidecar ─────────────────────────────────────
    meta_path = save_path / f"{prefix}_classes.json"
    meta_path.write_text(json.dumps({"class_names": class_names}, indent=2))

    print(f"\n{'='*65}")
    print(f"  Training complete!")
    print(f"  Best val_acc  : {best_val_acc:.4f}")
    print(f"  Best ckpt     : {best_ckpt_path}")
    print(f"  Class metadata: {meta_path}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m agents.tumor_classification_agent.train",
        description=(
            "Two-phase training for the Tumor Classification Agent.\n"
            "Phase 1: frozen backbone (heads only). "
            "Phase 2: last 30 layers unfrozen."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--stage", required=True, choices=["type", "grade"],
        help="'type' = Stage 1 ensemble (4 classes). 'grade' = Stage 2 grader (3 classes).",
    )
    p.add_argument(
        "--data-path", required=True,
        help=(
            "Stage 'type': path to data/tumor_classification/type/ "
            "(must contain Training/ and Testing/ sub-folders). "
            "Stage 'grade': path to data/tumor_classification/grade/ "
            "(must contain kaggle_3m/ sub-folder)."
        ),
    )
    p.add_argument(
        "--brats-path", default=None,
        help=(
            "[Grade stage only] Path to data/raw/BraTS2020_TrainingData/. "
            "Required when --stage grade."
        ),
    )
    p.add_argument(
        "--epochs", type=int, default=40,
        help="Total training epochs (Phase 1 = first 15, Phase 2 = rest). Default: 40.",
    )
    p.add_argument(
        "--save-dir", default="models/tumor_classifier/",
        help="Directory to write checkpoints. Default: models/tumor_classifier/",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Mini-batch size. Auto-selected if omitted (16 type / 8 grade).",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Phase-1 initial learning rate. Default: 1e-3.",
    )
    p.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers. Set 0 on Windows if multiprocessing issues occur. Default: 4.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42.",
    )
    p.add_argument(
        "--label-smoothing", type=float, default=0.1,
        help="Label smoothing ε. Default: 0.1.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
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
