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
from monai.data import CacheDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODALITY_KEYS = ["flair", "t1", "t1ce", "t2"]
ROI_SIZE = (128, 128, 128)
PIXDIM = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

from monai.transforms import Resized
def _shared_load_and_preprocess() -> list:
    """Base preprocessing pipeline."""
    return [
        LoadImaged(keys=MODALITY_KEYS, image_only=True),
        EnsureChannelFirstd(keys=MODALITY_KEYS),
        # Kaggle DICOM sequences have different shapes, force them to the same size before concatenating
        Resized(keys=MODALITY_KEYS, spatial_size=ROI_SIZE, mode="trilinear"),
        ConcatItemsd(keys=MODALITY_KEYS, name="image", dim=0),
        DeleteItemsd(keys=MODALITY_KEYS),
        Orientationd(keys=["image"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
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
            # Check if this is a UCSF-PDGM row or BraTS row
            if "BraTS21ID" in row and pd.notna(row["BraTS21ID"]):
                # Pandas might cast to float if there are NaNs, so convert to int first
                try:
                    subject_id = str(int(row["BraTS21ID"])).zfill(5)
                except ValueError:
                    subject_id = str(row["BraTS21ID"]).zfill(5)
                    
                subject_dir = self.data_dir / subject_id
                if not subject_dir.exists():
                    subject_dir = self.data_dir / f"BraTS2021_{subject_id}"
                idh_val = row.get("IDH_value", 0.0)
                mgmt_val = row.get("MGMT_value", 0.0)
            elif "ID" in row and pd.notna(row["ID"]): # UCSF format
                subject_id = str(row["ID"]) # e.g. UCSF-PDGM-0004
                # UCSF folders look like UCSF-PDGM-0004_nifti
                subject_dir = self.data_dir / "UCSF-PDGM" / "UCSF-PDGM" / f"{subject_id}_nifti"
                
                # Parse IDH: 'Mutant' -> 1.0, 'Wildtype' -> 0.0
                idh_raw = str(row.get("IDH", "")).lower()
                idh_val = 1.0 if "mutant" in idh_raw else 0.0
                
                # Parse MGMT: 'Methylated' -> 1.0, 'Unmethylated' -> 0.0
                mgmt_raw = str(row.get("MGMT status", "")).lower()
                mgmt_val = 1.0 if "methylated" in mgmt_raw and "unmethylated" not in mgmt_raw else 0.0
            else:
                continue

            if not subject_dir.exists():
                warnings.warn(f"Subject dir not found for {subject_id} at {subject_dir}")
                continue
                
            # Locate modalities (using rglob for nested UCSF folders)
            paths = {}
            dicom_mappings = {
                "flair": "FLAIR",
                "t1": "T1w",
                "t1ce": "T1wCE",
                "t2": "T2w"
            }
            # UCSF specific keywords
            ucsf_mappings = {
                "flair": "FLAIR",
                "t1": "T1_bias",
                "t1ce": "T1gad",
                "t2": "T2_bias"
            }

            for mod in MODALITY_KEYS:
                # 1. Try standard BraTS name directly
                matches = list(subject_dir.rglob(f"*{mod}*.nii.gz")) + list(subject_dir.rglob(f"*{mod}*.nii"))
                
                # 2. Try UCSF specific name if standard not found
                if not matches and "UCSF" in subject_id:
                    ucsf_mod = ucsf_mappings.get(mod, mod)
                    matches = list(subject_dir.rglob(f"*{ucsf_mod}*.nii.gz")) + list(subject_dir.rglob(f"*{ucsf_mod}*.nii"))

                if matches:
                    paths[mod] = str(matches[0])
                else:
                    dcm_dir = subject_dir / dicom_mappings.get(mod, mod)
                    if dcm_dir.exists() and dcm_dir.is_dir():
                        paths[mod] = str(dcm_dir)
            
            if len(paths) == 4:
                item = {
                    **paths, 
                    "idh": float(idh_val), 
                    "mgmt": float(mgmt_val)
                }
                self.samples.append(item)
            else:
                warnings.warn(f"Missing modalities for {subject_id}. Found: {list(paths.keys())}")
                
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

    Uses MONAI CacheDataset to preprocess and cache all 3D volumes in RAM
    after the first pass — eliminating the 200s-per-epoch DICOM loading
    bottleneck. Subsequent epochs run in ~15-20s instead of 200-250s.
    """
    full_dataset = RadiogenomicsDataset(csv_path, data_dir, transform=None)

    total = len(full_dataset)
    if total == 0:
        raise ValueError("No valid samples found. Check paths and CSV.")

    val_len   = int(total * val_split)
    train_len = total - val_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_len, val_len], generator=generator
    )

    # Build raw data dicts for CacheDataset (MONAI expects list-of-dicts)
    def _make_data_list(subset):
        items = []
        for i in subset.indices:
            s = full_dataset.samples[i]
            items.append({
                "flair":  s["flair"],
                "t1":     s["t1"],
                "t1ce":   s["t1ce"],
                "t2":     s["t2"],
                "labels": torch.tensor([s["idh"], s["mgmt"]], dtype=torch.float32),
            })
        return items

    train_data = _make_data_list(train_subset)
    val_data   = _make_data_list(val_subset)

    # MONAI CacheDataset: transforms run ONCE per sample, result cached in RAM.
    # num_workers for caching phase (parallelises the slow DICOM loading).
    # cache_rate=1.0 means ALL samples are cached.
    print(f"[Radiogenomics] Caching {len(train_data)} train + {len(val_data)} val volumes in RAM...")
    print("  (This takes ~2 minutes on first run, then epochs are instant)")

    # Wrap transforms to also pass through the pre-built labels tensor
    def _make_cached_dataset(data_list, is_train):
        img_transform = get_radiogenomics_transforms(is_train=is_train)

        class _LabelPassthrough(Compose):
            """Run image transforms; keep labels tensor unchanged."""
            def __call__(self, data):
                labels = data.pop("labels")
                out    = img_transform(data)
                out["labels"] = labels
                return out

        return CacheDataset(
            data=data_list,
            transform=_LabelPassthrough(transforms=[]),   # calls __call__ above
            cache_rate=1.0,
            num_workers=min(num_workers, 8),
            progress=True,
        )

    # Simpler: just use CacheDataset with the image transform then grab labels separately
    # Build a lightweight wrapper that CacheDataset can use
    class _RadioCacheDataset(torch.utils.data.Dataset):
        """Cache preprocessed volumes in RAM; labels stored separately."""
        def __init__(self, data_list, is_train):
            img_transform = get_radiogenomics_transforms(is_train=is_train)
            # Separate image dicts and labels
            img_dicts = []
            self._labels = []
            for item in data_list:
                lbl = item.pop("labels")
                img_dicts.append(item)
                self._labels.append(lbl)

            print(f"  Caching {'train' if is_train else 'val'} ({len(img_dicts)} volumes)...")
            self._cache = CacheDataset(
                data=img_dicts,
                transform=img_transform,
                cache_rate=1.0,
                num_workers=min(num_workers, 8),
                progress=True,
            )

        def __len__(self):
            return len(self._cache)

        def __getitem__(self, idx):
            data   = self._cache[idx]
            image  = data["image"]
            labels = self._labels[idx]
            return image, labels

    train_ds = _RadioCacheDataset(train_data, is_train=True)
    val_ds   = _RadioCacheDataset(val_data,   is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,       # 0 workers: cached data lives in main process RAM
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader
