"""
dataset.py
----------
DataLoaders for Stage 1 (tumor type) and Stage 2 (glioma grade) datasets.

Actual disk layout (verified):
-------------------------------
Stage 1 — Tumor Type
    data/tumor_classification/type/
        Training/  glioma/  meningioma/  notumor/  pituitary/   (1400 JPEGs each)
        Testing/   glioma/  meningioma/  notumor/  pituitary/   (400 JPEGs each)
    Classes: glioma | meningioma | notumor | pituitary
    Format : Standard torchvision ImageFolder.

Stage 2 — Glioma Grade
    grade_II source:
        data/tumor_classification/grade/kaggle_3m/
            <TCGA_patient_id>/  *.tif (MRI slices, 256x256 RGB)
                                *_mask.tif  (excluded -- only non-mask slices used)
        110 patient folders, ~20 slices each.

    grade_III / grade_IV source (BraTS 2020 -- H5 format):
        data/raw/BraTS2020_training_data/content/data/
            volume_N_slice_S.h5   (one file per 2-D slice)
        Each H5 file contains:
            image: (240, 240, 4) -- 4 MRI modalities (FLAIR, T1, T1ce, T2)
            mask:  (240, 240, 3) -- ch0=NCR/NET, ch1=Edema, ch2=ET (Enhancing Tumour)
        Total: 369 volumes x ~100 slices ~36,900 H5 files.

    Grade labeling (derived at dataset init from ET pixel count per volume):
        et_total = sum of mask channel-2 pixels across ALL slices of a volume
        Grade II   : et_total <    50  (matches LGG kaggle_3m patients)
        Grade III  : 50 <= et_total < 3000  (anaplastic glioma)
        Grade IV   : et_total >= 3000  (GBM -- majority of HGG volumes)

    Slice-level preprocessing:
        Use T1ce channel (image[:, :, 2]) -- best contrast for tumor boundary.
        Normalise to [0, 1] float32, repeat channel x 3 -> (3, H, W) RGB-like.
        Skip slices where ALL mask channels sum < 10 (background / no-tumour).

Both datasets:
    Full augmentation pipeline for training.
    Only resize + normalise for validation / inference.
    WeightedRandomSampler used to handle class imbalance.
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 224          # input resolution for all backbones

# Canonical class orders (determines integer label assignment)
TYPE_CLASSES  = ["glioma", "meningioma", "notumor", "pituitary"]
GRADE_CLASSES = ["grade_II", "grade_III", "grade_IV"]

# BraTS H5 grade thresholds (ET pixel count across all slices of a volume)
BRATS_ET_GRADE_II_MAX  =    50    # et_total < 50  -> grade_II
BRATS_ET_GRADE_III_MAX =  3000    # 50 <= et_total < 3000 -> grade_III
                                   # et_total >= 3000 -> grade_IV

# Minimum mask pixel sum to keep a slice (skip pure background)
BRATS_MIN_MASK_SUM = 10

# T1ce channel index inside the H5 image array (shape 240,240,4)
BRATS_T1CE_CHANNEL = 2


# ---------------------------------------------------------------------------
# Custom transforms
# ---------------------------------------------------------------------------

class EnsureRGB:
    """Convert any PIL mode to RGB (handles L, RGBA, I, F, P...).

    Placed first in every transform pipeline so all downstream ops receive
    a 3-channel tensor.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img


class AddGaussianNoise:
    """Add zero-mean Gaussian noise after ToTensor (values in [0, 1]).

    Args:
        std: Standard deviation; kept small relative to unit pixel range.
    """

    def __init__(self, std: float = 0.02) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def _train_transforms() -> transforms.Compose:
    """Full augmentation pipeline used during training."""
    return transforms.Compose([
        EnsureRGB(),
        # Resize slightly over target so random crop has room to move
        transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        # Zoom: RandomAffine with scale acts as zoom in/out
        transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.02),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _val_transforms() -> transforms.Compose:
    """Deterministic pipeline for validation and inference."""
    return transforms.Compose([
        EnsureRGB(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Weighted sampler helper
# ---------------------------------------------------------------------------

def _make_weighted_sampler(
    labels: List[int],
    num_classes: int,
) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that equalises class frequency."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weight_per_class = 1.0 / np.maximum(counts, 1.0)
    sample_weights   = torch.tensor(
        [weight_per_class[lbl] for lbl in labels], dtype=torch.float32
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ============================================================================
# Stage 1 — Tumor Type Dataset (ImageFolder)
# ============================================================================

def get_type_dataloaders(
    data_root:   str | Path,
    batch_size:  int  = 16,
    num_workers: int  = 4,
    val_split:   float = 0.15,
    seed:        int  = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train / val / test DataLoaders for Stage 1 (tumor type).

    Expected disk layout::

        <data_root>/
            Training/
                glioma/        *.jpg  (1400 images)
                meningioma/    *.jpg
                notumor/       *.jpg
                pituitary/     *.jpg
            Testing/
                glioma/        *.jpg  (400 images)
                meningioma/    *.jpg
                notumor/       *.jpg
                pituitary/     *.jpg

    A validation split is carved from Training using ``val_split`` fraction.
    The Testing folder is returned as a held-out test loader (no augmentation).

    Args:
        data_root:   Path to the type dataset root (contains Training/ Testing/).
        batch_size:  Mini-batch size. Use 16 for RTX 3050, 64 for DGX.
        num_workers: DataLoader worker count.
        val_split:   Fraction of Training to use as validation (default 0.15).
        seed:        Random seed for reproducibility.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_root = Path(data_root)
    train_dir = data_root / "Training"
    test_dir  = data_root / "Testing"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Testing directory not found: {test_dir}")

    # ── Build full training dataset ──────────────────────────────────────────
    # Use augmented transforms; we'll override val subset below
    full_train_aug = datasets.ImageFolder(
        root=str(train_dir),
        transform=_train_transforms(),
    )
    class_names = full_train_aug.classes
    n_classes   = len(class_names)

    # ── Stratified train/val split ───────────────────────────────────────────
    rng     = torch.Generator().manual_seed(seed)
    n_total = len(full_train_aug)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    train_idx, val_idx = torch.utils.data.random_split(
        range(n_total), [n_train, n_val], generator=rng
    )
    train_idx = list(train_idx)
    val_idx   = list(val_idx)

    # Val subset needs clean transforms — use a separately instantiated dataset
    full_train_clean = datasets.ImageFolder(
        root=str(train_dir),
        transform=_val_transforms(),
    )

    train_subset = torch.utils.data.Subset(full_train_aug,   train_idx)
    val_subset   = torch.utils.data.Subset(full_train_clean, val_idx)

    # ── Weighted sampler for training ────────────────────────────────────────
    train_labels = [full_train_aug.targets[i] for i in train_idx]
    sampler      = _make_weighted_sampler(train_labels, n_classes)

    # ── Test dataset (no augmentation) ───────────────────────────────────────
    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=_val_transforms(),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,          # mutually exclusive with shuffle=True
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    print(
        f"[TypeDataset] classes={class_names} | "
        f"train={n_train} val={n_val} test={len(test_dataset)} | "
        f"batch={batch_size}"
    )
    return train_loader, val_loader, test_loader, class_names


# ============================================================================
# Stage 2 — Glioma Grade Dataset (custom)
# ============================================================================

# ---------------------------------------------------------------------------
# Grade II — LGG kaggle_3m TIF reader
# ---------------------------------------------------------------------------

def _collect_grade_ii_samples(kaggle_3m_dir: Path) -> List[Path]:
    """Return paths to all non-mask MRI slices under kaggle_3m/.

    Each TCGA patient folder contains:
        TCGA_XXX_N.tif       ← MRI slice (keep)
        TCGA_XXX_N_mask.tif  ← segmentation mask (skip)

    Args:
        kaggle_3m_dir: Path to grade/kaggle_3m/

    Returns:
        List of .tif slice Paths (masks excluded).
    """
    slices = []
    for patient_dir in sorted(kaggle_3m_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        for tif in sorted(patient_dir.glob("*.tif")):
            if "_mask" not in tif.name:
                slices.append(tif)
    return slices


# ---------------------------------------------------------------------------
# Grade III / IV — BraTS NIfTI slice extractor
# ---------------------------------------------------------------------------

def _read_brats_grade_map(brats_train_dir: Path) -> Dict[str, str]:
    """Parse name_mapping.csv → {BraTS20_Training_XXX: 'grade_III'|'grade_IV'}.

    BraTS Grade column:
        HGG (High-Grade Glioma)  → grade_IV  (GBM, WHO grade IV)
        LGG (Low-Grade Glioma)   → grade_III  (WHO grade II/III;
                                    we assign grade_III as the dominant LGG
                                    grade in BraTS, distinct from the kaggle
                                    LGG which provides the grade_II class)

    Args:
        brats_train_dir: Path containing name_mapping.csv.

    Returns:
        dict mapping BraTS_2020_subject_ID → 'grade_III' or 'grade_IV'.
    """
    csv_path = brats_train_dir / BRATS_NAME_MAPPING
    if not csv_path.exists():
        raise FileNotFoundError(f"BraTS name_mapping.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    grade_map: Dict[str, str] = {}
    for _, row in df.iterrows():
        bid = row["BraTS_2020_subject_ID"]
        if not isinstance(bid, str):
            continue
        raw_grade = str(row["Grade"]).strip().upper()
        if raw_grade == "HGG":
            grade_map[bid] = "grade_IV"
        elif raw_grade == "LGG":
            grade_map[bid] = "grade_III"
    return grade_map


def _extract_flair_slices(
    nii_path:     Path,
    slice_low:    int   = BRATS_SLICE_LOW,
    slice_high:   int   = BRATS_SLICE_HIGH,
    min_int_frac: float = BRATS_MIN_INTENSITY_FRAC,
) -> List[np.ndarray]:
    """Extract informative 2D axial slices from a BraTS FLAIR NIfTI volume.

    Args:
        nii_path:     Path to *_flair.nii file.
        slice_low:    First axial slice index to consider.
        slice_high:   One-past-last axial slice index to consider.
        min_int_frac: Skip slices whose max < min_int_frac × volume max.

    Returns:
        List of uint8 numpy arrays of shape (H, W), one per kept slice.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "nibabel is required for BraTS NIfTI loading. "
            "Install with: pip install nibabel"
        )

    img   = nib.load(str(nii_path))
    vol   = img.get_fdata().astype(np.float32)   # (240, 240, 155)
    v_max = vol.max()
    if v_max == 0:
        return []

    threshold = min_int_frac * v_max
    slices: List[np.ndarray] = []
    hi = min(slice_high, vol.shape[2])

    for z in range(slice_low, hi):
        sl = vol[:, :, z]
        if sl.max() < threshold:
            continue
        # Normalise to [0, 255] uint8
        sl_norm = (sl / v_max * 255).astype(np.uint8)
        slices.append(sl_norm)

    return slices


def _collect_brats_samples(
    brats_train_dir: Path,
) -> Tuple[List[Path], List[str]]:
    """Walk the BraTS training directory and collect (flair_path, grade_label) pairs.

    Returns:
        (nii_paths, grades) — parallel lists of FLAIR .nii paths and grade strings.
    """
    grade_map = _read_brats_grade_map(brats_train_dir)
    nii_paths: List[Path] = []
    grades:    List[str]  = []

    for patient_id, grade_label in grade_map.items():
        patient_dir = brats_train_dir / patient_id
        if not patient_dir.exists():
            warnings.warn(f"BraTS patient dir missing: {patient_dir}", stacklevel=2)
            continue
        flair_nii = patient_dir / f"{patient_id}_flair.nii"
        if not flair_nii.exists():
            # Try .nii.gz
            flair_nii = patient_dir / f"{patient_id}_flair.nii.gz"
        if not flair_nii.exists():
            warnings.warn(f"FLAIR NIfTI not found: {patient_dir}", stacklevel=2)
            continue
        nii_paths.append(flair_nii)
        grades.append(grade_label)

    return nii_paths, grades


# ---------------------------------------------------------------------------
# GradeDataset — unified torch.utils.data.Dataset
# ---------------------------------------------------------------------------

class GradeDataset(Dataset):
    """Dataset for glioma grade classification (grade_II / grade_III / grade_IV).

    Sources:
        grade_II:   kaggle_3m TIF slices (256×256 RGB, loaded directly via PIL)
        grade_III:  BraTS LGG FLAIR NIfTI slices (2D axial, extracted at init)
        grade_IV:   BraTS HGG FLAIR NIfTI slices (2D axial, extracted at init)

    All samples are stored in memory as PIL Images after init (the dataset is
    small enough: ~2200 grade_II TIFs + ~6000 BraTS slices ≈ 330 MB at uint8).
    This avoids repeated NIfTI I/O during training.

    Args:
        kaggle_3m_dir:   Path to grade/kaggle_3m/
        brats_train_dir: Path to BraTS MICCAI_BraTS2020_TrainingData/
        transform:       torchvision transform applied to each PIL Image.
        grade_classes:   Ordered list of class names (determines integer labels).
        preload_brats:   If True (default), extract all BraTS slices at __init__
                         time. If False, slices are re-extracted per epoch (slower
                         but uses less RAM).
    """

    CLASS_NAMES = GRADE_CLASSES  # ["grade_II", "grade_III", "grade_IV"]

    def __init__(
        self,
        kaggle_3m_dir:    Path,
        brats_train_dir:  Path,
        transform:        Optional[Callable] = None,
        grade_classes:    List[str]          = GRADE_CLASSES,
        preload_brats:    bool               = True,
    ) -> None:
        super().__init__()
        self.transform     = transform
        self.class_to_idx  = {cls: i for i, cls in enumerate(grade_classes)}
        self.classes       = grade_classes

        # ── Collect grade_II samples (TIF paths) ─────────────────────────────
        g2_paths = _collect_grade_ii_samples(Path(kaggle_3m_dir))
        if not g2_paths:
            raise FileNotFoundError(
                f"No grade_II TIF slices found under {kaggle_3m_dir}"
            )

        # ── Collect grade_III/IV BraTS samples ───────────────────────────────
        nii_paths, nii_grades = _collect_brats_samples(Path(brats_train_dir))
        if not nii_paths:
            raise FileNotFoundError(
                f"No BraTS patients found under {brats_train_dir}"
            )

        # ── Build sample list: (PIL Image, label_int) ─────────────────────────
        self._images: List[Image.Image] = []
        self._labels: List[int]         = []

        # Grade II TIFs
        lbl_g2 = self.class_to_idx["grade_II"]
        print(f"[GradeDataset] Loading {len(g2_paths)} grade_II TIF slices...")
        for tif_path in g2_paths:
            try:
                img = Image.open(str(tif_path)).convert("RGB")
                self._images.append(img)
                self._labels.append(lbl_g2)
            except Exception as exc:
                warnings.warn(f"Failed to load {tif_path}: {exc}", stacklevel=2)

        # Grade III / IV BraTS slices
        g3_count = g4_count = 0
        if preload_brats:
            print(
                f"[GradeDataset] Extracting BraTS slices from "
                f"{len(nii_paths)} volumes (this may take ~1 min)..."
            )
            for nii_path, grade_label in zip(nii_paths, nii_grades):
                lbl = self.class_to_idx[grade_label]
                try:
                    slices = _extract_flair_slices(nii_path)
                except Exception as exc:
                    warnings.warn(f"Failed to read {nii_path}: {exc}", stacklevel=2)
                    continue
                for sl_arr in slices:
                    img = Image.fromarray(sl_arr, mode="L").convert("RGB")
                    self._images.append(img)
                    self._labels.append(lbl)
                    if grade_label == "grade_III":
                        g3_count += 1
                    else:
                        g4_count += 1

        print(
            f"[GradeDataset] Loaded: "
            f"grade_II={len(g2_paths)} | "
            f"grade_III={g3_count} | "
            f"grade_IV={g4_count} | "
            f"total={len(self._images)}"
        )

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img   = self._images[idx]
        label = self._labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def targets(self) -> List[int]:
        """Integer label list, compatible with WeightedRandomSampler helpers."""
        return self._labels


# ---------------------------------------------------------------------------
# Grade DataLoader factory
# ---------------------------------------------------------------------------

def get_grade_dataloaders(
    grade_root:     str | Path,
    brats_root:     str | Path,
    batch_size:     int   = 8,
    num_workers:    int   = 4,
    val_split:      float = 0.15,
    seed:           int   = 42,
    preload_brats:  bool  = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Build train / validation DataLoaders for Stage 2 (glioma grade).

    Args:
        grade_root:    Path to data/tumor_classification/grade/
                       (must contain kaggle_3m/ sub-folder).
        brats_root:    Path to data/raw/BraTS2020_TrainingData/
                       (must contain MICCAI_BraTS2020_TrainingData/).
        batch_size:    Mini-batch size. Use 8 for RTX 3050, 64 for DGX.
        num_workers:   DataLoader worker count. Set 0 on Windows if errors occur.
        val_split:     Fraction of data to use as validation.
        seed:          Random seed.
        preload_brats: Extract all BraTS slices at startup (recommended).

    Returns:
        (train_loader, val_loader, class_names)
    """
    grade_root = Path(grade_root)
    brats_root = Path(brats_root)

    kaggle_3m_dir   = grade_root / "kaggle_3m"
    brats_train_dir = brats_root / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"

    if not kaggle_3m_dir.exists():
        raise FileNotFoundError(f"kaggle_3m dir not found: {kaggle_3m_dir}")
    if not brats_train_dir.exists():
        raise FileNotFoundError(f"BraTS training dir not found: {brats_train_dir}")

    # ── Build full dataset with augmentation ─────────────────────────────────
    full_aug = GradeDataset(
        kaggle_3m_dir=kaggle_3m_dir,
        brats_train_dir=brats_train_dir,
        transform=_train_transforms(),
        preload_brats=preload_brats,
    )

    n_total = len(full_aug)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    rng = torch.Generator().manual_seed(seed)
    train_idx, val_idx = torch.utils.data.random_split(
        range(n_total), [n_train, n_val], generator=rng
    )
    train_idx = list(train_idx)
    val_idx   = list(val_idx)

    # ── Val subset with clean transforms ─────────────────────────────────────
    # Build a second GradeDataset with val transforms
    # (shares the same preloaded images list via a lightweight wrapper)
    full_clean = GradeDataset(
        kaggle_3m_dir=kaggle_3m_dir,
        brats_train_dir=brats_train_dir,
        transform=_val_transforms(),
        preload_brats=preload_brats,
    )

    train_subset = torch.utils.data.Subset(full_aug,   train_idx)
    val_subset   = torch.utils.data.Subset(full_clean, val_idx)

    # ── Weighted sampler ──────────────────────────────────────────────────────
    train_labels = [full_aug.targets[i] for i in train_idx]
    sampler      = _make_weighted_sampler(train_labels, len(GRADE_CLASSES))

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    class_names = full_aug.classes
    print(
        f"[GradeDataset] Loaders: train={n_train} val={n_val} | "
        f"batch={batch_size} | classes={class_names}"
    )
    return train_loader, val_loader, class_names


# ============================================================================
# TTA transforms (used by inference.py)
# ============================================================================

def get_tta_transforms(n_augmentations: int = 10) -> List[transforms.Compose]:
    """Return a list of deterministic transforms for Test-Time Augmentation.

    Index 0 is always the clean / canonical view.  Remaining entries are
    fixed augmentations (specific flip/rotation parameters, not random).

    Args:
        n_augmentations: Total TTA views including the clean view (max 10).

    Returns:
        List of transforms.Compose objects, length == n_augmentations.
    """
    def _base(extra: List) -> transforms.Compose:
        return transforms.Compose([
            EnsureRGB(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            *extra,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    variants = [
        _base([]),                                                              # 0: clean
        _base([transforms.RandomHorizontalFlip(p=1.0)]),                       # 1: hflip
        _base([transforms.RandomVerticalFlip(p=1.0)]),                         # 2: vflip
        _base([transforms.RandomRotation(degrees=(10, 10))]),                  # 3: rot+10
        _base([transforms.RandomRotation(degrees=(-10, -10))]),                # 4: rot-10
        _base([transforms.RandomRotation(degrees=(15, 15))]),                  # 5: rot+15
        _base([transforms.RandomRotation(degrees=(-15, -15))]),                # 6: rot-15
        _base([transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
               transforms.CenterCrop(IMAGE_SIZE)]),                            # 7: zoom-in
        _base([transforms.ColorJitter(brightness=(1.15, 1.30))]),              # 8: bright+
        _base([transforms.ColorJitter(brightness=(0.70, 0.85))]),              # 9: bright-
    ]

    n = max(1, min(n_augmentations, len(variants)))
    return variants[:n]
