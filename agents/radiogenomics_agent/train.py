import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
"""
train.py
--------
Training script for the Radiogenomics Agent (IDH and MGMT prediction).
Uses PyTorch Automatic Mixed Precision (AMP) to maximize VRAM efficiency.

Fixes applied:
  - Early stopping (patience=10) to stop before overfitting
  - Saves best checkpoint by val_acc (not val_loss)
  - Updated AMP API (torch.amp instead of torch.cuda.amp)
  - Gradient clipping for stability
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from agents.radiogenomics_agent.model import build_radiogenomics_model
from agents.radiogenomics_agent.dataset import get_radiogenomics_dataloaders


def _calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate binary accuracy for IDH and MGMT (threshold = 0.5 on sigmoid)."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == targets).float().mean().item()


def train(
    csv_path:    str,
    data_dir:    str,
    epochs:      int,
    batch_size:  int,
    lr:          float,
    num_workers: int,
    save_dir:    str,
    patience:    int  = 10,
    seed:        int  = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Radiogenomics Agent Training | Device: {device}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Patience: {patience}")
    print(f"{'='*65}\n")

    # 1. Dataloaders (with RAM caching)
    print("[1/3] Building dataloaders (caching volumes to RAM — takes ~2 min)...")
    train_loader, val_loader = get_radiogenomics_dataloaders(
        csv_path=csv_path,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=0.2,
        seed=seed,
    )

    # 2. Model, Loss, Optimizer, AMP
    print("[2/3] Initializing model and AMP...")
    model = build_radiogenomics_model(device=device)

    # Multi-label binary cross-entropy
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler    = GradScaler("cuda")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5,  # mode='max' for val_acc
    )

    print("[3/3] Starting training loop...\n")
    best_val_acc    = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t_start = time.perf_counter()

        # ── Training Phase ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_acc  = 0.0
        n_train    = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits = model(images)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_acc  += _calculate_accuracy(logits, labels)
            n_train    += 1

        train_loss /= n_train
        train_acc  /= n_train

        # ── Validation Phase ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        n_val    = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast("cuda"):
                    logits = model(images)
                    loss   = criterion(logits, labels)

                val_loss += loss.item()
                val_acc  += _calculate_accuracy(logits, labels)
                n_val    += 1

        if n_val > 0:
            val_loss /= n_val
            val_acc  /= n_val
        else:
            val_loss = float('inf')
            val_acc  = 0.0

        scheduler.step(val_acc)   # scheduler watches val_acc now
        t_end = time.perf_counter()

        print(f"Ep {epoch:3d}/{epochs} | "
              f"tr_loss: {train_loss:.4f} tr_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} | "
              f"{(t_end - t_start):.1f}s")

        # ── Checkpoint on val_acc improvement ───────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = save_path / "radiogenomics_best.pth"
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":        val_loss,
                "val_acc":         val_acc,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path.name}  (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⏹  Early stopping triggered at epoch {epoch}  "
                      f"(no val_acc improvement for {patience} epochs)")
                print(f"  Best val_acc: {best_val_acc:.4f}")
                break

    print(f"\nTraining complete.  Best val_acc = {best_val_acc:.4f}")


def _build_parser():
    p = argparse.ArgumentParser(description="Train Radiogenomics 3D CNN")
    p.add_argument("--csv-path",    type=str, required=True, help="Path to labels.csv")
    p.add_argument("--data-dir",    type=str, required=True, help="Directory containing patient subfolders")
    p.add_argument("--epochs",      type=int,   default=100,  help="Max training epochs (early stopping may end sooner)")
    p.add_argument("--batch-size",  type=int,   default=4,    help="Batch size (4 is safe for 3D volumes on DGX)")
    p.add_argument("--lr",          type=float, default=1e-4, help="Learning rate")
    p.add_argument("--num-workers", type=int,   default=0,    help="Dataloader workers")
    p.add_argument("--save-dir",    type=str,   default="models/radiogenomics/", help="Save directory")
    p.add_argument("--patience",    type=int,   default=10,   help="Early stopping patience (epochs without improvement)")
    p.add_argument("--seed",        type=int,   default=42,   help="Random seed")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(
        csv_path    = args.csv_path,
        data_dir    = args.data_dir,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        num_workers = args.num_workers,
        save_dir    = args.save_dir,
        patience    = args.patience,
        seed        = args.seed,
    )
