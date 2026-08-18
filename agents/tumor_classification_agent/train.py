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
from torch.amp import GradScaler, autocast
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

PHASE1_EPOCHS   = 25      # epochs with frozen backbone
PHASE2_LR_SCALE = 0.05   # phase-2 backbone LR = phase-1 LR * 0.05
N_UNFREEZE      = 10000   # Set massive to safely unfreeze ENTIRE backbone for >95% clinical accuracy


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


def _freeze_backbone_bn(model: nn.Module) -> None:
    """Stabilise ALL backbone BatchNorm layers during training.

    Two behaviours depending on whether the BN layer is frozen or unfrozen:

    Frozen BN (weight.requires_grad=False)
        Keep in eval() mode entirely — running stats stay at ImageNet values.
        Same strategy as Phase 1.

    Unfrozen BN (weight.requires_grad=True, i.e. inside the fine-tuned blocks)
        Remain in train() mode so the BN CAN adapt to MRI data, BUT reduce
        momentum from 0.1 to 0.01.  With batch_size=16 and the default
        momentum=0.1 the running stats drift by up to ±0.3 per epoch
        (random walk from noisy mini-batches) causing the pituitary/glioma
        oscillation seen in Phase 2 fine-tuning.  Momentum=0.01 gives a
        10× slower EMA: statistics stabilise over ~100 batches instead of ~10.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None and not module.weight.requires_grad:
                # Fully frozen backbone layer — lock running stats
                module.eval()
            else:
                # Unfrozen layer — allow adaptation but slow the EMA
                module.momentum = 0.01

def _train_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    optimizer:   optim.Optimizer,
    device:      torch.device,
    scaler:      GradScaler,
    is_ensemble: bool,
    mixup_alpha: float = 0.0,
) -> Tuple[float, float]:
    """Run one full training epoch.

    Ensemble loss uses AVERAGED LOGITS (not sum of individual CE losses).
    Averaging logits before CE forces all three backbones to cooperate:
    none can minimise loss alone by predicting the easy 'notumor' class.

    When mixup_alpha > 0, Mixup augmentation is applied to 50% of batches.
    Mixup smooths the glioma/meningioma decision boundary by creating convex
    combinations of training pairs — the primary reason glioma accuracy is
    stuck at 80-83% without it.

    Returns:
        (mean_loss, mean_accuracy) over all batches.
    """
    import random
    import numpy as np

    model.train()
    # Stabilise BN: freeze fully-frozen layers; slow momentum on unfrozen ones.
    _freeze_backbone_bn(model)
    total_loss = total_acc = 0.0
    n_batches  = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Mixup (applied to 50% of batches when alpha > 0) ──────────────
        use_mixup = mixup_alpha > 0 and random.random() < 0.5
        if use_mixup:
            lam  = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx  = torch.randperm(images.size(0), device=device)
            images = lam * images + (1.0 - lam) * images[idx]
            labels_b = labels[idx]

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            if is_ensemble:
                assert isinstance(model, TumorTypeEnsemble)
                eff_l, res_l, den_l = model.individual_logits(images)
                avg_l = (eff_l + res_l + den_l) / 3.0
                if use_mixup:
                    loss = lam * criterion(avg_l, labels) + (1.0 - lam) * criterion(avg_l, labels_b)
                else:
                    loss = criterion(avg_l, labels)
                with torch.no_grad():
                    probs = model(images)
            else:
                logits = model(images)
                if use_mixup:
                    loss = lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, labels_b)
                else:
                    loss = criterion(logits, labels)
                probs = F.softmax(logits.detach(), dim=1)

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
    """Run one full validation epoch in full float32 (no autocast).

    autocast is deliberately omitted here: the ensemble runs three separate
    forward passes whose float16 logits can overflow to inf, producing
    NaN val_loss. Since there is no backward pass, precision matters more
    than speed.

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

        # Full float32 — avoids NaN from float16 overflow in ensemble logits
        if is_ensemble:
            assert isinstance(model, TumorTypeEnsemble)
            eff_l, res_l, den_l = model.individual_logits(images)
            # Averaged logits — consistent with training loss
            avg_l = (eff_l + res_l + den_l) / 3.0
            loss  = criterion(avg_l.float(), labels)
            probs = model(images)   # averaged softmax for accuracy
        else:
            logits = model(images)
            loss   = criterion(logits.float(), labels)
            probs  = F.softmax(logits.float(), dim=1)

        # Guard: skip NaN batches rather than propagating corruption
        loss_val = loss.item()
        if not torch.isfinite(loss).item():
            loss_val = 0.0

        total_loss += loss_val
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
    stage:        str,
    data_path:    str,
    epochs:       int            = 40,
    save_dir:     str            = "models/tumor_classifier/",
    batch_size:   Optional[int]  = None,
    lr:           float          = 1e-3,
    num_workers:  int            = 4,
    seed:         int            = 42,
    smoothing:    float          = 0.1,
    brats_h5_dir: Optional[str]  = None,
    resume_from:  Optional[str]  = None,
    mixup_alpha:  float          = 0.0,
) -> None:
    """Full two-phase training run for one stage.

    Args:
        stage:        "type" (Stage 1) or "grade" (Stage 2).
        data_path:    For type: path to type/ root (Training/ + Testing/).
                      For grade: path to grade/ root (kaggle_3m/).
        epochs:       Total epochs for a fresh run OR Phase-2 extra epochs
                      when --resume is set.
        save_dir:     Directory where checkpoints are saved.
        batch_size:   Mini-batch size; auto-selected if None (16 type, 8 grade).
        lr:           Phase-1 initial LR (fresh run) or Phase-2 backbone LR
                      base when resuming (actual LR = lr * PHASE2_LR_SCALE).
        num_workers:  DataLoader worker count.
        seed:         Random seed.
        smoothing:    Label-smoothing factor.
        brats_h5_dir: (Grade stage only) path to BraTS H5 directory
                      (contains volume_N_slice_S.h5 files).
                      e.g. data/raw/BraTS2020_training_data/content/data/
        resume_from:  Path to a saved .pth checkpoint.  When set, Phase 1 is
                      skipped and Phase 2 runs for ``epochs`` more epochs
                      starting from the loaded weights.
        mixup_alpha:  Beta distribution alpha for Mixup augmentation (0=off).
                      Recommended: 0.4 for small medical datasets.
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
        bs = batch_size or 256
        train_loader, _val_loader, test_loader, class_names = get_type_dataloaders(
            data_root=data_path,
            batch_size=bs,
            num_workers=num_workers,
            seed=seed,
            val_split=0.0,   # use ALL 5600 Training/ images; Testing/ is our val set
        )
        # Use the official Testing/ split as validation — not a random carve from
        # Training/.  This gives 5600 training images (vs 4760) and a proper
        # held-out 1600-image val set, reducing overfitting and giving honest metrics.
        val_loader  = test_loader
        model       = build_type_ensemble(num_classes=len(class_names), device=device)
        is_ensemble = True
        prefix      = "type_ensemble"

    else:  # grade
        if brats_h5_dir is None:
            raise ValueError(
                "--brats-path is required for --stage grade.\n"
                "Example: --brats-path data/raw/BraTS2020_training_data/content/data/"
            )
        bs = batch_size or 256
        train_loader, val_loader, class_names = get_grade_dataloaders(
            grade_root=data_path,
            brats_root=brats_h5_dir,
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
    scaler    = GradScaler("cuda")
    best_val_acc   = 0.0
    best_ckpt_path = save_path / f"{prefix}_best.pth"

    # ── Optional: resume from checkpoint ────────────────────────────────────
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_val_acc = ckpt.get("val_accuracy", 0.0)
        print(f"[resume] Loaded weights from '{resume_from}'")
        print(f"[resume] Resuming from val_acc={best_val_acc:.4f}  — skipping Phase 1")
        print(f"[resume] Running {epochs} more Phase-2 epochs\n")

    # ── PHASE 1 — frozen backbone ────────────────────────────────────────────
    phase1_eps = min(PHASE1_EPOCHS, epochs)
    if resume_from is not None:
        # Skip Phase 1 entirely when resuming
        phase1_eps = 0
    else:
        model.set_phase(1)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=phase1_eps, eta_min=lr * 1e-3)

        _SEP = "─" * 65
        print("\n" + _SEP)
        print(f"  PHASE 1  |  Frozen backbone  |  {phase1_eps} epochs  |  LR={lr:.0e}")
        print(_SEP)

        for epoch in range(1, phase1_eps + 1):
            t0 = time.time()
            tr_loss, tr_acc = _train_epoch(
                model, train_loader, criterion, optimizer, device, scaler,
                is_ensemble, mixup_alpha=0.0,   # no mixup in Phase 1 — let heads converge first
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
                f"  Ep {epoch:3d}/{phase1_eps} \u2502 "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} \u2502 "
                f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} \u2502 "
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
    phase2_eps = epochs if resume_from is not None else (epochs - phase1_eps)
    if phase2_eps > 0:
        phase2_lr  = lr * PHASE2_LR_SCALE

        model.set_phase(2, n_unfreeze=N_UNFREEZE)

        # Differential LR: newly unfrozen conv layers (pretrained ImageNet weights)
        # need a much lower LR than the heads (randomly initialised).
        # backbone LR = phase2_lr (5e-5),  head LR = phase2_lr * 5 (2.5e-4)
        if is_ensemble:
            head_ids = set()
            for backbone, head_attr in [
                (model.efficientnet_b4, "classifier"),
                (model.resnet50,        "fc"),
                (model.densenet121,     "classifier"),
            ]:
                head_ids.update(id(p) for p in getattr(backbone, head_attr).parameters())
            backbone_p = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
            head_p     = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
            optimizer  = optim.AdamW([
                {"params": backbone_p, "lr": phase2_lr},
                {"params": head_p,     "lr": phase2_lr * 5},
            ], weight_decay=1e-4)
        else:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=phase2_lr, weight_decay=1e-4,
            )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=phase2_eps,
            eta_min=phase2_lr * 1e-2,   # floor at 5e-7; avoids LR starvation
        )

        print(f"\n{'─'*65}")
        label = f"RESUMED from {Path(resume_from).name}" if resume_from else f"Last {N_UNFREEZE} layers unfrozen"
        print(f"  PHASE 2  |  {label}  |  {phase2_eps} epochs  |  backbone LR={phase2_lr:.0e}")
        print(f"{'─'*65}")

        start_ep = 1
        end_ep   = phase2_eps

        for epoch in range(start_ep, end_ep + 1):
            t0 = time.time()
            tr_loss, tr_acc = _train_epoch(
                model, train_loader, criterion, optimizer, device, scaler,
                is_ensemble, mixup_alpha=mixup_alpha,
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
                f"  Ep {epoch:3d}/{end_ep} \u2502 "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} \u2502 "
                f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} \u2502 "
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
        "--brats-path", default=None, dest="brats_h5_dir",
        help=(
            "[Grade stage only] Path to BraTS H5 directory containing "
            "volume_N_slice_S.h5 files. "
            "e.g. data/raw/BraTS2020_training_data/content/data/ "
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
        "--batch-size", type=int, default=256,
        help="Mini-batch size. Default: 256 (optimized for RTX 6000 Ada).",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Phase-1 initial learning rate. Default: 1e-3.",
    )
    p.add_argument(
        "--num-workers", type=int, default=16,
        help="DataLoader workers. Set 0 on Windows if multiprocessing issues occur. Default: 16.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42.",
    )
    p.add_argument(
        "--resume", default=None, metavar="CHECKPOINT",
        help=(
            "Path to a .pth checkpoint to resume from. "
            "Loads model weights, skips Phase 1, and runs Phase 2 for "
            "--epochs more epochs at LR = --lr * PHASE2_LR_SCALE. "
            "Example: --resume models/tumor_classifier/type_ensemble_best.pth"
        ),
    )
    p.add_argument(
        "--mixup", type=float, default=0.4, metavar="ALPHA",
        help=(
            "Mixup Beta distribution alpha for Phase 2 augmentation. "
            "0 = disabled. 0.4 = recommended for small medical datasets. Default: 0.4."
        ),
    )
    p.add_argument(
        "--label-smoothing", type=float, default=0.1,
        help="Label smoothing eps (0=standard CE, 0.1=default).",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(
        stage        = args.stage,
        data_path    = args.data_path,
        epochs       = args.epochs,
        save_dir     = args.save_dir,
        batch_size   = args.batch_size,
        lr           = args.lr,
        num_workers  = args.num_workers,
        seed         = args.seed,
        smoothing    = args.label_smoothing,
        brats_h5_dir = args.brats_h5_dir,
        resume_from  = args.resume,
        mixup_alpha  = args.mixup,
    )
