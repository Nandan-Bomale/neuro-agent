"""
dataset.py
----------
DataLoaders for Stage 1 (tumor type) and Stage 2 (glioma grade) datasets.

Stage 1 — Tumor Type
    Source  : data/tumor_classification/type/
    Classes : glioma | meningioma | pituitary | no_tumor
    Format  : JPEG images organised in sub-folders per class (ImageFolder layout).

Stage 2 — Glioma Grade
    Source  : data/tumor_classification/grade/   (Grade II, LGG dataset)
              data/raw/                            (Grade III / IV BraTS 2020 slices)
    Classes : grade_II | grade_III | grade_IV
    Format  : ImageFolder layout.

Both datasets:
    • Grayscale MRI → 3-channel RGB by replicating the single channel.
    • Full augmentation pipeline for training.
    • Only resize + normalise for validation / inference.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224          # Input resolution for all models

# Type-stage class order (must match folder names)
TYPE_CLASSES  = ["glioma", "meningioma", "no_tumor", "pituitary"]
GRADE_CLASSES = ["grade_II", "grade_III", "grade_IV"]


# ---------------------------------------------------------------------------
# Custom transforms
# ---------------------------------------------------------------------------

class GrayscaleToRGB:
    """Replicate a single-channel image to produce a 3-channel tensor.

    Works on PIL Images.  If the image already has 3 channels, it is returned
    unchanged.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode == "L":
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return img


class AddGaussianNoise:
    """Add zero-mean Gaussian noise to a tensor (applied after ToTensor).

    Args:
        std: Standard deviation of the Gaussian noise.  Kept small relative
             to the [0, 1] pixel range produced by ToTensor.
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
        GrayscaleToRGB(),
        transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),  # slight oversize for crop
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),  # zoom ±15 %
        transforms.ToTensor(),
        AddGaussianNoise(std=0.02),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _val_transforms() -> transforms.Compose:
    """Deterministic pipeline for validation and inference."""
    return transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Weighted sampler (handles class imbalance)
# ---------------------------------------------------------------------------

def _build_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that up-samples minority classes."""
    class_counts = np.bincount([s[1] for s in dataset.samples])
    weights_per_class = 1.0 / np.maximum(class_counts, 1)
    sample_weights = np.array([weights_per_class[label] for _, label in dataset.samples])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ---------------------------------------------------------------------------
# Public API — DataLoader factories
# ---------------------------------------------------------------------------

def get_type_dataloaders(
    data_root: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, list]:
    """Build train / validation DataLoaders for Stage 1 (tumor type).

    Args:
        data_root:   Root directory containing per-class sub-folders
                     (glioma/, meningioma/, no_tumor/, pituitary/).
        batch_size:  Mini-batch size.  Use 16 for RTX 3050, 64 for DGX.
        num_workers: Number of DataLoader worker processes.
        val_split:   Fraction of data to use for validation.
        seed:        Random seed for reproducibility.

    Returns:
        (train_loader, val_loader, class_names)
    """
    data_root = Path(data_root)

    # Load full dataset with training augmentations first to get class info
    full_dataset = datasets.ImageFolder(root=str(data_root), transform=_train_transforms())
    class_names  = full_dataset.classes

    # Stratified split
    torch.manual_seed(seed)
    n_total = len(full_dataset)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # Override transforms for validation split (no augmentation)
    val_dataset   = datasets.ImageFolder(root=str(data_root), transform=_val_transforms())
    val_subset_no_aug = torch.utils.data.Subset(val_dataset, val_subset.indices)

    # Weighted sampler for training (handles imbalanced classes)
    train_tmp = datasets.ImageFolder(root=str(data_root))
    train_samples = [train_tmp.samples[i] for i in train_subset.indices]

    class_counts = np.bincount([s[1] for s in train_samples], minlength=len(class_names))
    weights_per_class = 1.0 / np.maximum(class_counts, 1)
    sample_weights = np.array([weights_per_class[s[1]] for s in train_samples])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset_no_aug,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"[TypeDataset] Classes: {class_names} | "
        f"Train={n_train} | Val={n_val} | Batch={batch_size}"
    )
    return train_loader, val_loader, class_names


def get_grade_dataloaders(
    data_root: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, list]:
    """Build train / validation DataLoaders for Stage 2 (glioma grade).

    Args:
        data_root:   Root directory containing per-class sub-folders
                     (grade_II/, grade_III/, grade_IV/).
        batch_size:  Mini-batch size.  Use 8 for RTX 3050, 64 for DGX.
        num_workers: Number of DataLoader worker processes.
        val_split:   Fraction of data to use for validation.
        seed:        Random seed for reproducibility.

    Returns:
        (train_loader, val_loader, class_names)
    """
    data_root = Path(data_root)

    full_dataset = datasets.ImageFolder(root=str(data_root), transform=_train_transforms())
    class_names  = full_dataset.classes

    torch.manual_seed(seed)
    n_total = len(full_dataset)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    val_dataset       = datasets.ImageFolder(root=str(data_root), transform=_val_transforms())
    val_subset_no_aug = torch.utils.data.Subset(val_dataset, val_subset.indices)

    train_tmp = datasets.ImageFolder(root=str(data_root))
    train_samples = [train_tmp.samples[i] for i in train_subset.indices]
    class_counts  = np.bincount([s[1] for s in train_samples], minlength=len(class_names))
    weights_per_class = 1.0 / np.maximum(class_counts, 1)
    sample_weights    = np.array([weights_per_class[s[1]] for s in train_samples])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset_no_aug,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"[GradeDataset] Classes: {class_names} | "
        f"Train={n_train} | Val={n_val} | Batch={batch_size}"
    )
    return train_loader, val_loader, class_names


# ---------------------------------------------------------------------------
# TTA transform helper (used by inference.py)
# ---------------------------------------------------------------------------

def get_tta_transforms(n_augmentations: int = 10) -> list:
    """Return a list of deterministic transforms for Test-Time Augmentation.

    Each transform in the list represents one augmented view of the input.
    The first entry is always the clean / canonical view.

    Args:
        n_augmentations: Total number of TTA views (including the clean view).

    Returns:
        List of ``transforms.Compose`` objects, length == n_augmentations.
    """
    clean = _val_transforms()

    tta_list = [clean]  # index 0: no augmentation

    aug_variants = [
        transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.RandomRotation(degrees=(-10, -10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(brightness=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(brightness=-0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomAffine(degrees=0, scale=(1.1, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomAffine(degrees=0, scale=(0.9, 0.9)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    ]

    # Fill up to n_augmentations (cap at available variants + clean)
    while len(tta_list) < n_augmentations and len(tta_list) <= len(aug_variants):
        tta_list.append(aug_variants[len(tta_list) - 1])

    return tta_list[:n_augmentations]
