"""
dataset.py
----------
Radiogenomics Dataset and DataLoaders for IDH and MGMT prediction.
Expects a 3D MRI volume (4 standard BraTS modalities) and a labels.csv file.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from monai.transforms import (
    Compose,
    ConcatItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    SpatialPadd,
    DeleteItemsd,
    CenterSpatialCropd,
    RandSpatialCropd,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODALITY_KEYS = ["flair", "t1", "t1ce", "t2"]
ROI_SIZE = (128, 128, 128)
PIXDIM = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _shared_load_and_preprocess() -> list:
    """Base preprocessing pipeline."""
    return [
        LoadImaged(keys=MODALITY_KEYS, image_only=True),
        EnsureChannelFirstd(keys=MODALITY_KEYS),
        ConcatItemsd(keys=MODALITY_KEYS, name="image", dim=0),
        DeleteItemsd(keys=MODALITY_KEYS),
        Spacingd(keys=["image"], pixdim=PIXDIM, mode=("bilinear",)),
        Orientationd(keys=["image"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=ROI_SIZE, mode="constant"),
    ]

def get_radiogenomics_transforms(is_train: bool = True) -> Compose:
    """Builds MONAI transform pipeline for 3D Radiogenomics."""
    transforms = _shared_load_and_preprocess()

    if is_train:
        transforms.extend([
            RandSpatialCropd(keys=["image"], roi_size=ROI_SIZE, random_center=True, random_size=False),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image"], prob=0.1, max_k=3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ])
    else:
        transforms.append(
            CenterSpatialCropd(keys=["image"], roi_size=ROI_SIZE)
        )
    
    transforms.append(EnsureTyped(keys=["image"], dtype="float32"))
    return Compose(transforms)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RadiogenomicsDataset(Dataset):
    """
    Dataset that loads 4 NIfTI modalities per patient and yields
    a tuple: (image_tensor, target_tensor)
    
    target_tensor shape: [2] -> [IDH_value, MGMT_value]
    """
    def __init__(
        self, 
        csv_path: str | Path, 
        data_dir: str | Path, 
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        print(f"[RadiogenomicsDataset] Loading labels from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        self.samples = []
        for _, row in self.df.iterrows():
            subject_id = str(row["BraTS21ID"])
            # In BraTS21, IDs are often zero-padded to 5 digits, e.g. "00001"
            subject_id = subject_id.zfill(5)
            
            subject_dir = self.data_dir / subject_id
            if not subject_dir.exists():
                subject_dir = self.data_dir / f"BraTS2021_{subject_id}"
            
            if not subject_dir.exists():
                warnings.warn(f"Subject dir not found for {subject_id}")
                continue
                
            # Locate modalities
            paths = {}
            dicom_mappings = {
                "flair": "FLAIR",
                "t1": "T1w",
                "t1ce": "T1wCE",
                "t2": "T2w"
            }
            for mod in MODALITY_KEYS:
                matches = list(subject_dir.glob(f"*{mod}*.nii.gz"))
                if not matches:
                    matches = list(subject_dir.glob(f"*{mod}*.nii"))
                
                if matches:
                    paths[mod] = str(matches[0])
                else:
                    dcm_dir = subject_dir / dicom_mappings.get(mod, mod)
                    if dcm_dir.exists() and dcm_dir.is_dir():
                        paths[mod] = str(dcm_dir)
            
            if len(paths) == 4:
                item = {
                    **paths, 
                    "idh": float(row.get("IDH_value", 0.0)), 
                    "mgmt": float(row.get("MGMT_value", 0.0))
                }
                self.samples.append(item)
            else:
                warnings.warn(f"Missing modalities for {subject_id}")
                
        print(f"[RadiogenomicsDataset] Found {len(self.samples)} complete samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        data_dict = {
            "flair": item["flair"],
            "t1": item["t1"],
            "t1ce": item["t1ce"],
            "t2": item["t2"],
        }
        
        if self.transform:
            data_dict = self.transform(data_dict)
            
        image = data_dict["image"]
        # Float targets for BCEWithLogitsLoss
        labels = torch.tensor([item["idh"], item["mgmt"]], dtype=torch.float32)
        return image, labels


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def get_radiogenomics_dataloaders(
    csv_path:    str | Path, 
    data_dir:    str | Path, 
    batch_size:  int   = 8, 
    num_workers: int   = 4,
    val_split:   float = 0.2,
    seed:        int   = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits the dataset and returns train and validation dataloaders.
    """
    dataset = RadiogenomicsDataset(csv_path, data_dir, transform=None)
    
    total = len(dataset)
    if total == 0:
        raise ValueError("No valid samples found. Check paths and CSV.")
        
    val_len = int(total * val_split)
    train_len = total - val_len
    
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_len, val_len], generator=generator
    )
    
    # Wrapper to lazily apply transforms AFTER random_split
    class TransformWrap(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __len__(self):
            return len(self.subset)
            
        def __getitem__(self, idx):
            # Access underlying sample directly to avoid tuple slicing issues
            real_idx = self.subset.indices[idx]
            item = self.subset.dataset.samples[real_idx]
            
            data_dict = {
                "flair": item["flair"],
                "t1": item["t1"],
                "t1ce": item["t1ce"],
                "t2": item["t2"],
            }
            if self.transform:
                data_dict = self.transform(data_dict)
                
            image = data_dict["image"]
            labels = torch.tensor([item["idh"], item["mgmt"]], dtype=torch.float32)
            return image, labels

    train_ds_wrapped = TransformWrap(train_ds, get_radiogenomics_transforms(is_train=True))
    val_ds_wrapped = TransformWrap(val_ds, get_radiogenomics_transforms(is_train=False))
    
    train_loader = DataLoader(
        train_ds_wrapped, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds_wrapped, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader
