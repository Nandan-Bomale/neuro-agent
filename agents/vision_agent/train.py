import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
"""
train.py
--------
Training script for the BraTS 2020 Vision Agent U-Net.

Features
--------
  - Dice + Cross-Entropy loss (DiceCELoss)
  - Sliding-window inference for validation (handles full 3-D volumes)
  - Dice metric tracked per epoch; best model checkpoint saved automatically
  - AdamW optimiser with cosine-annealing LR schedule
  - TensorBoard logging (loss + Dice per epoch)
  - Configurable from the CLI or by importing run_training()

Hardware note
-------------
  Default settings (batch_size=2, roi=128³) are tuned for the RTX 3050
  (4 GB VRAM).  Training 50 epochs on 277 cases will take ~6-12 hours on
  the laptop.  Send the same script to the DGX to finish in <1 hour.

Usage — CLI
-----------
    python -m agents.vision_agent.train

Usage — import
--------------
    from agents.vision_agent.train import run_training
    run_training(max_epochs=50, batch_size=2)
"""

import os
import time
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from agents.vision_agent.dataset import get_dataloaders
from agents.vision_agent.model import (
    build_loss,
    build_metric,
    build_post_transforms,
    build_unet,
    count_parameters,
)
from agents.vision_agent.transforms import ROI_SIZE

# ---------------------------------------------------------------------------
# Training defaults — override via run_training() kwargs or CLI args
# ---------------------------------------------------------------------------

DATA_ROOT       = "c:/Neuro Agent/data/raw"
CHECKPOINT_DIR  = "c:/Neuro Agent/models/vision"
LOG_DIR         = "c:/Neuro Agent/models/vision/logs"

MAX_EPOCHS      = 50
BATCH_SIZE      = 2
NUM_WORKERS     = 0       # 0 = main process only (no multiprocessing).
                          # Required on Windows to avoid worker OOM crashes.
                          # Set to 2-4 on Linux/DGX for faster data loading.
LR              = 1e-4    # AdamW learning rate
WEIGHT_DECAY    = 1e-5    # L2 regularisation
VAL_INTERVAL    = 2       # run validation every N epochs (saves time)

# Sliding-window inference settings for validation
SW_OVERLAP      = 0.5     # 50% overlap between windows (standard for BraTS)
SW_MODE         = "gaussian"  # gaussian weighting reduces stitching artefacts


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def run_training(
    data_root:      str   = DATA_ROOT,
    checkpoint_dir: str   = CHECKPOINT_DIR,
    log_dir:        str   = LOG_DIR,
    max_epochs:     int   = MAX_EPOCHS,
    batch_size:     int   = BATCH_SIZE,
    num_workers:    int   = NUM_WORKERS,
    lr:             float = LR,
    weight_decay:   float = WEIGHT_DECAY,
    val_interval:   int   = VAL_INTERVAL,
    resume:         bool  = False,
) -> dict:
    """Train the Vision Agent U-Net on BraTS 2020.

    Saves the best model checkpoint (by validation Dice) to:
        <checkpoint_dir>/best_model.pth

    Also saves a rolling latest checkpoint after every epoch to:
        <checkpoint_dir>/latest_checkpoint.pth
    Use ``resume=True`` (or ``--resume`` on CLI) to continue training
    from the last completed epoch after a power cut or crash.

    Also saves the final checkpoint at the end of training:
        <checkpoint_dir>/final_model.pth

    Args:
        data_root:      Path to directory containing BraTS2020_TrainingData/.
        checkpoint_dir: Directory to save model checkpoints.
        log_dir:        Directory for TensorBoard logs.
        max_epochs:     Total number of training epochs.
        batch_size:     Number of 128³ patches per training step.
                        Use 2 for RTX 3050 (4 GB VRAM).
        num_workers:    DataLoader worker processes.
        lr:             AdamW learning rate.
        weight_decay:   AdamW weight decay (L2 regularisation).
        val_interval:   Run validation every this many epochs.
        resume:         If True, load latest_checkpoint.pth and continue
                        training from the saved epoch.  No-op if no
                        checkpoint exists yet.

    Returns:
        dict with keys:
            best_val_dice  : float — best validation Dice score achieved
            best_epoch     : int   — epoch at which best Dice was achieved
            best_ckpt_path : str   — path to the saved best checkpoint
    """
    # ── Setup ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    print("=" * 60)
    print("NeuroAgent -- Vision Agent Training")
    print("=" * 60)
    print(f"Device         : {device}")
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU            : {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")
    print(f"Max epochs     : {max_epochs}")
    print(f"Batch size     : {batch_size}")
    print(f"Learning rate  : {lr}")
    print(f"ROI size       : {ROI_SIZE}")
    print(f"Val interval   : every {val_interval} epochs")
    print(f"Resume         : {resume}")
    print(f"Checkpoints    : {checkpoint_dir}")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────────────────────
    print("\n[1/4] Building dataloaders...")
    train_loader, val_loader = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    print("\n[2/4] Building model...")
    model   = build_unet(device)
    loss_fn = build_loss()
    metric  = build_metric()
    post_pred, post_label = build_post_transforms()
    print(f"Parameters     : {count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")

    # ── Optimiser & Scheduler ───────────────────────────────────────────────
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs, eta_min=lr * 0.01)

    # ── Resume from checkpoint ──────────────────────────────────────────────
    start_epoch    = 1
    best_val_dice  = -1.0
    best_epoch     = -1
    best_ckpt_path = str(Path(checkpoint_dir) / "best_model.pth")
    latest_ckpt    = str(Path(checkpoint_dir) / "latest_checkpoint.pth")

    if resume and Path(latest_ckpt).exists():
        print(f"\n[RESUME] Loading checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optimiser_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch   = ckpt["epoch"] + 1       # resume from NEXT epoch
        best_val_dice = ckpt.get("best_val_dice", -1.0)
        best_epoch    = ckpt.get("best_epoch",    -1)
        print(f"[RESUME] Resuming from epoch {start_epoch}/{max_epochs}")
        print(f"[RESUME] Best val Dice so far: {best_val_dice:.4f} (epoch {best_epoch})")
    elif resume:
        print("[RESUME] No checkpoint found — starting from scratch.")

    # ── Training Loop ───────────────────────────────────────────────────────
    print("\n[3/4] Starting training...\n")

    for epoch in range(start_epoch, max_epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        epoch_loss   = 0.0
        train_steps  = 0
        t_epoch_start = time.time()

        for batch in train_loader:
            # CacheDataset with RandCropByPosNegLabeld returns a list of dicts
            # (num_samples patches per volume).  Stack them into a single batch.
            if isinstance(batch, list):
                inputs = torch.cat([b["image"] for b in batch], dim=0).to(device)
                labels = torch.cat([b["label"] for b in batch], dim=0).to(device)
            else:
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)

            optimiser.zero_grad()
            logits = model(inputs)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimiser.step()

            epoch_loss  += loss.item()
            train_steps += 1

        scheduler.step()

        avg_train_loss = epoch_loss / max(train_steps, 1)
        elapsed        = time.time() - t_epoch_start

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # ── Save latest checkpoint (overwrites each epoch) ───────────────────
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_dice":   best_val_dice,
                "best_epoch":      best_epoch,
            },
            latest_ckpt,
        )

        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # ── Validation ───────────────────────────────────────────────────────
        if epoch % val_interval == 0:
            model.eval()
            metric.reset()

            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)

                    # Sliding-window inference — processes the full 3-D volume
                    # in 128³ patches and stitches the output back together.
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs,
                        roi_size=ROI_SIZE,
                        sw_batch_size=batch_size,
                        predictor=model,
                        overlap=SW_OVERLAP,
                        mode=SW_MODE,
                    )

                    # Binarise predictions and compute Dice
                    val_outputs_bin = post_pred(val_outputs)
                    metric(y_pred=val_outputs_bin, y=val_labels)

            val_dice = metric.aggregate().item()
            metric.reset()

            writer.add_scalar("Dice/val", val_dice, epoch)

            print(f"  --> Val Dice : {val_dice:.4f}", end="")

            # Save best model
            if val_dice > best_val_dice:
                best_val_dice  = val_dice
                best_epoch     = epoch
                torch.save(
                    {
                        "epoch":          epoch,
                        "model_state":    model.state_dict(),
                        "optimiser_state": optimiser.state_dict(),
                        "val_dice":       val_dice,
                    },
                    best_ckpt_path,
                )
                print(f"  *** NEW BEST — saved to {best_ckpt_path}")
            else:
                print()

    # ── Save final checkpoint ────────────────────────────────────────────────
    print("\n[4/4] Saving final checkpoint...")
    final_ckpt_path = str(Path(checkpoint_dir) / "final_model.pth")
    torch.save(
        {
            "epoch":           max_epochs,
            "model_state":     model.state_dict(),
            "optimiser_state": optimiser.state_dict(),
            "val_dice":        best_val_dice,
        },
        final_ckpt_path,
    )

    writer.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best Val Dice  : {best_val_dice:.4f}")
    print(f"  Best Epoch     : {best_epoch}")
    print(f"  Best Checkpoint: {best_ckpt_path}")
    print("=" * 60)

    return {
        "best_val_dice":  best_val_dice,
        "best_epoch":     best_epoch,
        "best_ckpt_path": best_ckpt_path,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the NeuroAgent Vision Agent U-Net on BraTS 2020"
    )
    parser.add_argument("--data_root",      type=str,   default=DATA_ROOT)
    parser.add_argument("--checkpoint_dir", type=str,   default=CHECKPOINT_DIR)
    parser.add_argument("--log_dir",        type=str,   default=LOG_DIR)
    parser.add_argument("--max_epochs",     type=int,   default=MAX_EPOCHS)
    parser.add_argument("--batch_size",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--num_workers",    type=int,   default=NUM_WORKERS)
    parser.add_argument("--lr",             type=float, default=LR)
    parser.add_argument("--weight_decay",   type=float, default=WEIGHT_DECAY)
    parser.add_argument("--val_interval",   type=int,   default=VAL_INTERVAL)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from latest_checkpoint.pth if it exists.",
    )

    args = parser.parse_args()
    run_training(**vars(args))
