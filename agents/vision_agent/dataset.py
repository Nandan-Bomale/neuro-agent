"""
dataset.py
----------
BraTS 2020 dataset builder for the Vision Agent.

Responsibilities
----------------
  - Walk the BraTS 2020 folder and collect all valid case directories.
  - Build per-sample data dictionaries with keys: flair, t1, t1ce, t2, label.
  - Split cases into train / validation / test sets (deterministic seed).
  - Return MONAI DataLoader objects ready for training and evaluation.

Expected folder layout (BraTS 2020 Kaggle download)
-----------------------------------------------------
  <data_root>/
  └── BraTS2020_TrainingData/
      └── MICCAI_BraTS2020_TrainingData/
          ├── BraTS20_Training_001/
          │   ├── BraTS20_Training_001_flair.nii
          │   ├── BraTS20_Training_001_t1.nii
          │   ├── BraTS20_Training_001_t1ce.nii
          │   ├── BraTS20_Training_001_t2.nii
          │   └── BraTS20_Training_001_seg.nii
          ├── BraTS20_Training_002/ ...
          └── ...

Usage
-----
    from agents.vision_agent.dataset import get_dataloaders

    train_loader, val_loader = get_dataloaders(
        data_root="c:/Neuro Agent/data/raw",
        batch_size=2,
        num_workers=2,
    )
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from monai.data import CacheDataset, DataLoader, Dataset

from agents.vision_agent.transforms import (
    MODALITY_KEYS,
    get_train_transforms,
    get_val_transforms,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Sub-path from data_root to the training cases folder.
_TRAIN_SUBPATH = os.path.join(
    "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData"
)

#: Fraction of cases used for validation.
_VAL_FRACTION = 0.15

#: Fraction of cases used for testing (held-out, never seen during training).
_TEST_FRACTION = 0.10

#: Random seed for the train/val/test split — fix this so every run gets the
#: same split, making results reproducible and comparable across experiments.
_SPLIT_SEED = 42

#: Fraction of the dataset to cache in RAM using MONAI CacheDataset.
#: Set to 0.0 on 16 GB RAM laptops — caching all 277 BraTS cases needs
#: ~10 GB RAM which causes OOM crashes on Windows.
#: With cache_rate=0.0, each case is loaded fresh from disk each epoch
#: (slower, but stable).  On a DGX with 64+ GB RAM, set to 1.0 for
#: maximum speed.
_CACHE_RATE = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_sample_dict(case_dir: Path) -> Optional[Dict[str, str]]:
    """Build a MONAI-compatible data dictionary for a single BraTS case.

    Expects exactly 5 files (flair, t1, t1ce, t2, seg) in *case_dir*.
    Returns None if any expected file is missing (skips the case with a
    warning rather than crashing the whole run).

    Args:
        case_dir: Path — directory for one BraTS case, e.g.
                  .../MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/

    Returns:
        dict with keys: flair, t1, t1ce, t2, label  (all str paths)
        or None if any file is missing.
    """
    case_id = case_dir.name          # e.g. "BraTS20_Training_001"

    # Map modality key → expected filename suffix
    # Note: the Kaggle BraTS 2020 download stores uncompressed .nii files,
    # not the .nii.gz files found in the official Synapse release.
    suffix_map = {
        "flair": f"{case_id}_flair.nii",
        "t1":    f"{case_id}_t1.nii",
        "t1ce":  f"{case_id}_t1ce.nii",
        "t2":    f"{case_id}_t2.nii",
        "label": f"{case_id}_seg.nii",
    }

    sample = {}
    for key, filename in suffix_map.items():
        filepath = case_dir / filename
        if not filepath.exists():
            print(f"[dataset] WARNING: missing file {filepath} — skipping {case_id}")
            return None
        sample[key] = str(filepath)

    return sample


def _collect_all_samples(data_root: str) -> List[Dict[str, str]]:
    """Scan the BraTS 2020 training directory and return all valid samples.

    Args:
        data_root: str — root data directory (contains BraTS2020_TrainingData/).

    Returns:
        Sorted list of sample dicts, one per valid case.

    Raises:
        FileNotFoundError: if the expected BraTS directory does not exist.
    """
    train_dir = Path(data_root) / _TRAIN_SUBPATH

    if not train_dir.exists():
        raise FileNotFoundError(
            f"BraTS 2020 training data not found at: {train_dir}\n"
            f"Expected layout: {data_root}/{_TRAIN_SUBPATH}/BraTS20_Training_001/..."
        )

    # Gather all subdirectories that look like BraTS cases
    case_dirs = sorted([
        d for d in train_dir.iterdir()
        if d.is_dir() and d.name.startswith("BraTS20_Training_")
    ])

    samples = []
    for case_dir in case_dirs:
        sample = _build_sample_dict(case_dir)
        if sample is not None:
            samples.append(sample)

    print(f"[dataset] Found {len(samples)} valid cases in {train_dir}")
    return samples


def _split_samples(
    samples: List[Dict],
    val_fraction: float = _VAL_FRACTION,
    test_fraction: float = _TEST_FRACTION,
    seed: int = _SPLIT_SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Randomly split samples into train / val / test sets.

    The split is deterministic given the same seed — guarantees reproducibility
    across machines and runs.

    Args:
        samples:      Full list of sample dicts.
        val_fraction: Fraction of samples for validation.
        test_fraction: Fraction of samples for test (held-out).
        seed:         Random seed.

    Returns:
        Tuple (train_samples, val_samples, test_samples).
    """
    shuffled = samples.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_test  = max(1, int(n_total * test_fraction))
    n_val   = max(1, int(n_total * val_fraction))

    test_samples  = shuffled[:n_test]
    val_samples   = shuffled[n_test : n_test + n_val]
    train_samples = shuffled[n_test + n_val:]

    print(
        f"[dataset] Split -> train: {len(train_samples)} | "
        f"val: {len(val_samples)} | test: {len(test_samples)} cases"
    )
    return train_samples, val_samples, test_samples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_root: str = "c:/Neuro Agent/data/raw",
    batch_size: int = 2,
    num_workers: int = 2,
    cache_rate: float = _CACHE_RATE,
    val_fraction: float = _VAL_FRACTION,
    test_fraction: float = _TEST_FRACTION,
    seed: int = _SPLIT_SEED,
) -> Tuple[DataLoader, DataLoader]:
    """Build and return train and validation DataLoaders for BraTS 2020.

    Uses MONAI CacheDataset to load and preprocess each volume once, then
    serve cached tensors in subsequent epochs — significantly faster than
    re-reading NIfTI files from disk every epoch.

    Args:
        data_root:     Path to the directory containing BraTS2020_TrainingData/.
        batch_size:    Number of volumes per batch.  With ROI_SIZE=(128,128,128)
                       and the 4-channel input, batch_size=2 uses ~1.5 GB VRAM.
        num_workers:   Parallel workers for DataLoader.  2 is safe on most
                       laptops; reduce to 0 if you hit multiprocessing issues
                       on Windows.
        cache_rate:    Fraction of dataset to cache in RAM (0.0–1.0).
        val_fraction:  Fraction of cases for validation.
        test_fraction: Fraction of cases for test (not returned here; held out).
        seed:          Seed for the train/val/test split.

    Returns:
        Tuple (train_loader, val_loader) — both are MONAI DataLoader instances.
    """
    all_samples = _collect_all_samples(data_root)

    train_samples, val_samples, _ = _split_samples(
        all_samples, val_fraction, test_fraction, seed
    )

    # ── Training dataset ────────────────────────────────────────────────────
    # CacheDataset pre-runs all deterministic transforms (load, normalise,
    # crop) once.  The random augmentations run fresh each epoch because they
    # are appended after the cached portion inside get_train_transforms().
    train_ds = CacheDataset(
        data=train_samples,
        transform=get_train_transforms(),
        cache_rate=cache_rate,
        num_workers=num_workers,
        progress=True,
    )

    # ── Validation dataset ──────────────────────────────────────────────────
    # Full volumes, no random cropping — sliding-window inference in train.py
    # will handle the patch extraction during validation.
    val_ds = CacheDataset(
        data=val_samples,
        transform=get_val_transforms(),
        cache_rate=cache_rate,
        num_workers=num_workers,
        progress=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,         # faster CPU→GPU transfer
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,            # validation: one full volume at a time
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_samples(
    data_root: str = "c:/Neuro Agent/data/raw",
    val_fraction: float = _VAL_FRACTION,
    test_fraction: float = _TEST_FRACTION,
    seed: int = _SPLIT_SEED,
) -> List[Dict[str, str]]:
    """Return the held-out test sample dicts (paths only, no transforms applied).

    Used by evaluation scripts to run sliding-window inference on completely
    unseen cases and report final Dice scores.

    Args:
        data_root:     Path containing BraTS2020_TrainingData/.
        val_fraction:  Must match the value used during training.
        test_fraction: Must match the value used during training.
        seed:          Must match the value used during training.

    Returns:
        List of sample dicts for the test split.
    """
    all_samples = _collect_all_samples(data_root)
    _, _, test_samples = _split_samples(
        all_samples, val_fraction, test_fraction, seed
    )
    return test_samples
