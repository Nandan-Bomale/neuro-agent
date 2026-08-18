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
        transforms.Resize((IMAGE_SIZE + 30, IMAGE_SIZE + 30)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        # Zoom and shear
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2), shear=15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        AddGaussianNoise(std=0.03),
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
# Stage 2 - Glioma Grade Dataset (custom)
# ============================================================================

# ---------------------------------------------------------------------------
# Grade II - LGG kaggle_3m TIF reader
# ---------------------------------------------------------------------------

def _collect_grade_ii_samples(kaggle_3m_dir: Path) -> list:
    '''Return paths to all non-mask MRI slices under kaggle_3m/.

    Each TCGA patient folder contains:
        TCGA_XXX_N.tif       <- MRI slice (keep)
        TCGA_XXX_N_mask.tif  <- segmentation mask (skip)

    Args:
        kaggle_3m_dir: Path to grade/kaggle_3m/

    Returns:
        List of .tif slice Paths (masks excluded).
    '''
    slices = []
    for patient_dir in sorted(kaggle_3m_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        for tif in sorted(patient_dir.glob('*.tif')):
            if '_mask' not in tif.name:
                slices.append(tif)
    return slices


# ---------------------------------------------------------------------------
# Grade III / IV - BraTS 2020 H5 slice reader
# ---------------------------------------------------------------------------

def _parse_vol_idx(stem: str):
    '''Extract volume index N from filename stem 'volume_N_slice_S'.

    Returns None if the stem does not match the expected pattern.
    '''
    parts = stem.split('_')
    # Expected: ['volume', N, 'slice', S]
    if len(parts) >= 4 and parts[0] == 'volume' and parts[2] == 'slice':
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


def _compute_volume_grades(h5_dir: Path) -> dict:
    '''Scan all H5 files once and accumulate ET pixel counts per volume.

    For each volume N (from filename), reads mask[:, :, 2] (Enhancing
    Tumour channel) and accumulates the pixel sum.

    Grade thresholds (applied after summing across ALL slices of a volume):
        et_total <    50  -> 0  (grade_II)
        50 <= et_total < 3000 -> 1  (grade_III)
        et_total >= 3000  -> 2  (grade_IV)

    Args:
        h5_dir: Directory containing volume_N_slice_S.h5 files.

    Returns:
        Dict mapping volume_index (int) -> grade label integer (0/1/2).
    '''
    try:
        import h5py
    except ImportError:
        raise ImportError('h5py is required. pip install h5py')
    
    from collections import defaultdict as _dd

    h5_files = sorted(h5_dir.glob('volume_*_slice_*.h5'))
    if not h5_files:
        raise FileNotFoundError(
            f'No volume_*_slice_*.h5 files found in: {h5_dir}'
        )

    print(f'[BraTSH5] Scanning {len(h5_files)} H5 files for grade labeling ...')

    et_sums = _dd(int)
    for h5_path in h5_files:
        vol_idx = _parse_vol_idx(h5_path.stem)
        if vol_idx is None:
            continue
        try:
            with h5py.File(str(h5_path), 'r') as f:
                et_sums[vol_idx] += int(f['mask'][:, :, 2].sum())
        except Exception as exc:
            import warnings as _w
            _w.warn(f'Mask read error {h5_path.name}: {exc}', stacklevel=2)

    vol_to_grade = {}
    for vol_idx, et_total in et_sums.items():
        if et_total < BRATS_ET_GRADE_II_MAX:
            vol_to_grade[vol_idx] = 0   # grade_II
        elif et_total < BRATS_ET_GRADE_III_MAX:
            vol_to_grade[vol_idx] = 1   # grade_III
        else:
            vol_to_grade[vol_idx] = 2   # grade_IV

    c = [sum(1 for v in vol_to_grade.values() if v == i) for i in range(3)]
    print(f'[BraTSH5] Volumes: grade_II={c[0]} | grade_III={c[1]} | grade_IV={c[2]}')
    return vol_to_grade


class BraTSH5GradeDataset(Dataset):
    '''BraTS 2020 H5 slice dataset for glioma grade classification.

    Each H5 file is one 2-D axial slice from a 3-D MRI volume:
        image : (240, 240, 4) float64  - FLAIR | T1 | T1ce | T2
        mask  : (240, 240, 3) uint8    - NCR   | Edema | ET

    Grade labeling (via _compute_volume_grades, computed once at init):
        et_total < 50        -> grade_II   (label 0)
        50 <= et_total < 3000 -> grade_III  (label 1)
        et_total >= 3000     -> grade_IV   (label 2)

    Slices with mask.sum() < BRATS_MIN_MASK_SUM are skipped (background).

    __getitem__ pipeline:
        1. Open H5 file, read image[:, :, 2]  (T1ce channel)
        2. Normalise to [0, 1] float32
        3. Scale to [0, 255] uint8 -> grayscale PIL -> convert to RGB
        4. Apply self.transform -> (3, 224, 224) tensor

    Args:
        h5_dir:        Directory with volume_N_slice_S.h5 files.
        transform:     torchvision transform applied in __getitem__.
        grade_classes: Ordered list of class names; sets label integers.
        vol_to_grade:  Pre-computed {vol_idx: label_int} map.
                       If None, _compute_volume_grades is called at init.
    '''

    CLASS_NAMES = GRADE_CLASSES

    def __init__(
        self,
        h5_dir:        Path,
        transform:     Optional[Callable]       = None,
        grade_classes: List[str]                = GRADE_CLASSES,
        vol_to_grade:  Optional[Dict[int, int]] = None,
    ) -> None:
        super().__init__()
        try:
            import h5py
            self._h5py = h5py
        except ImportError:
            raise ImportError('h5py is required: pip install h5py')

        self.h5_dir       = Path(h5_dir)
        self.transform    = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(grade_classes)}
        self.classes      = grade_classes

        if vol_to_grade is None:
            vol_to_grade = _compute_volume_grades(self.h5_dir)
        self._vol_to_grade = vol_to_grade

        # Build (path, label_int) sample list, skipping background slices
        self._samples = []
        skipped_bg = skipped_unknown = 0
        all_h5 = sorted(self.h5_dir.glob('volume_*_slice_*.h5'))

        print(f'[BraTSH5] Building sample list from {len(all_h5)} H5 files ...')
        for h5_path in all_h5:
            vol_idx = _parse_vol_idx(h5_path.stem)
            if vol_idx is None:
                continue
            label = self._vol_to_grade.get(vol_idx)
            if label is None or label >= len(grade_classes):
                skipped_unknown += 1
                continue
            try:
                with self._h5py.File(str(h5_path), 'r') as f:
                    mask_sum = int(f['mask'][:].sum())
            except Exception as exc:
                warnings.warn(f'Cannot read {h5_path.name}: {exc}', stacklevel=2)
                continue
            if mask_sum < BRATS_MIN_MASK_SUM:
                skipped_bg += 1
                continue
            self._samples.append((h5_path, label))

        per = [sum(1 for _, l in self._samples if l == i)
               for i in range(len(grade_classes))]
        print(
            f'[BraTSH5] Kept {len(self._samples)} slices '
            f'(bg_skipped={skipped_bg} unknown={skipped_unknown})'
        )
        print('[BraTSH5] Per grade: ' +
              ' | '.join(f'{grade_classes[i]}={per[i]}'
                         for i in range(len(grade_classes))))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        h5_path, label = self._samples[idx]

        with self._h5py.File(str(h5_path), 'r') as f:
            image = f['image'][:]       # (240, 240, 4) float64

        # T1ce channel (index 2)
        t1ce = image[:, :, BRATS_T1CE_CHANNEL].astype(np.float32)

        t1ce_max = float(t1ce.max())
        if t1ce_max > 0.0:
            t1ce = t1ce / t1ce_max

        pil_img = Image.fromarray(
            (t1ce * 255).astype(np.uint8), mode='L'
        ).convert('RGB')

        if self.transform is not None:
            pil_img = self.transform(pil_img)

        return pil_img, label

    @property
    def targets(self) -> List[int]:
        return [lbl for _, lbl in self._samples]


# ---------------------------------------------------------------------------
# GradeDataset - kaggle TIF (grade_II) + BraTS H5 (grade_III / grade_IV)
# ---------------------------------------------------------------------------

class GradeDataset(Dataset):
    '''Unified glioma grade dataset.

    Combines:
        grade_II   : kaggle_3m TIF slices (loaded into RAM at init).
        grade_III  : BraTS H5 slices (lazy H5 read per __getitem__).
        grade_IV   : BraTS H5 slices (lazy H5 read per __getitem__).

    The vol_to_grade map is computed once and shared between train/val
    GradeDataset instances so 36,900 H5 files are only scanned once.

    Args:
        kaggle_3m_dir: Path to grade/kaggle_3m/
        h5_dir:        Path to directory with volume_N_slice_S.h5 files.
        transform:     Applied to every sample in __getitem__.
        grade_classes: Ordered list of grade class names.
        vol_to_grade:  Optional pre-computed {vol_idx: label_int} map.
    '''

    CLASS_NAMES = GRADE_CLASSES

    def __init__(
        self,
        kaggle_3m_dir: Path,
        h5_dir:        Path,
        transform:     Optional[Callable]       = None,
        grade_classes: List[str]                = GRADE_CLASSES,
        vol_to_grade:  Optional[Dict[int, int]] = None,
    ) -> None:
        super().__init__()
        self.transform    = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(grade_classes)}
        self.classes      = grade_classes

        # -- Grade II: TIF images into RAM -------------------------------------
        g2_paths = _collect_grade_ii_samples(Path(kaggle_3m_dir))
        if not g2_paths:
            raise FileNotFoundError(
                f'No grade_II TIF slices found under {kaggle_3m_dir}'
            )
        lbl_g2 = self.class_to_idx['grade_II']
        print(f'[GradeDataset] Loading {len(g2_paths)} grade_II TIF slices ...')
        self._tif_images = []
        self._tif_labels = []
        for tif_path in g2_paths:
            try:
                self._tif_images.append(Image.open(str(tif_path)).convert('RGB'))
                self._tif_labels.append(lbl_g2)
            except Exception as exc:
                warnings.warn(f'Failed to load {tif_path}: {exc}', stacklevel=2)

        # -- Grade III / IV: BraTS H5, lazy IO --------------------------------
        if vol_to_grade is None:
            vol_to_grade = _compute_volume_grades(Path(h5_dir))

        # Only pass grade_III (1) and grade_IV (2) volumes to BraTSH5GradeDataset
        g3_lbl = self.class_to_idx['grade_III']
        g4_lbl = self.class_to_idx['grade_IV']
        h5_vol_map = {v: lbl for v, lbl in vol_to_grade.items()
                      if lbl in (g3_lbl, g4_lbl)}

        self._h5_dataset = BraTSH5GradeDataset(
            h5_dir=Path(h5_dir),
            transform=None,
            grade_classes=grade_classes,
            vol_to_grade=h5_vol_map,
        )

        self._n_tif = len(self._tif_images)
        self._n_h5  = len(self._h5_dataset)
        print(
            f'[GradeDataset] Combined: '
            f'grade_II(TIF)={self._n_tif} | '
            f'grade_III+IV(H5)={self._n_h5} | '
            f'total={self._n_tif + self._n_h5}'
        )

    def __len__(self) -> int:
        return self._n_tif + self._n_h5

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx < self._n_tif:
            img   = self._tif_images[idx]
            label = self._tif_labels[idx]
        else:
            img, label = self._h5_dataset[idx - self._n_tif]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def targets(self) -> List[int]:
        return list(self._tif_labels) + self._h5_dataset.targets


# ---------------------------------------------------------------------------
# Grade DataLoader factory
# ---------------------------------------------------------------------------

def get_grade_dataloaders(
    grade_root:    str | Path,
    brats_root:    str | Path,
    batch_size:    int   = 8,
    num_workers:   int   = 4,
    val_split:     float = 0.15,
    seed:          int   = 42,
    preload_brats: bool  = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    '''Build train / validation DataLoaders for Stage 2 (glioma grade).

    Args:
        grade_root:    Path to data/tumor_classification/grade/
                       Must contain kaggle_3m/ with grade_II TIF files.
        brats_root:    Path to the DIRECTORY that directly contains H5 files.
                       e.g. data/raw/BraTS2020_training_data/content/data/
                       NO sub-directory is appended - point straight at the H5s.
        batch_size:    Mini-batch size. 8 for RTX 3050, 32 for DGX.
        num_workers:   Workers. 0 on Windows, 4+ on Linux/DGX.
        val_split:     Fraction held out for validation (default 0.15).
        seed:          Random seed.
        preload_brats: Ignored - kept for API backward-compatibility.

    Returns:
        (train_loader, val_loader, class_names)
    '''
    grade_root = Path(grade_root)
    h5_dir     = Path(brats_root)     # no sub-directory appended
    kaggle_3m  = grade_root / 'kaggle_3m'

    if not kaggle_3m.exists():
        raise FileNotFoundError(f'kaggle_3m dir not found: {kaggle_3m}')
    if not h5_dir.exists():
        raise FileNotFoundError(f'BraTS H5 dir not found: {h5_dir}')

    # Compute grade map ONCE - shared between train + val to avoid double scan
    vol_to_grade = _compute_volume_grades(h5_dir)

    full_aug = GradeDataset(
        kaggle_3m_dir=kaggle_3m,
        h5_dir=h5_dir,
        transform=_train_transforms(),
        vol_to_grade=vol_to_grade,
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

    full_clean = GradeDataset(
        kaggle_3m_dir=kaggle_3m,
        h5_dir=h5_dir,
        transform=_val_transforms(),
        vol_to_grade=vol_to_grade,   # reuse - no extra H5 scan
    )

    train_subset = torch.utils.data.Subset(full_aug,   train_idx)
    val_subset   = torch.utils.data.Subset(full_clean, val_idx)

    all_targets  = full_aug.targets
    train_labels = [all_targets[i] for i in train_idx]
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
        f'[GradeDataset] Loaders ready: train={n_train} val={n_val} | '
        f'batch={batch_size} | classes={class_names}'
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
