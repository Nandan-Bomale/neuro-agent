"""
transforms.py
-------------
MONAI transform pipelines for BraTS 2020 brain MRI data.

Three pipelines are provided:
  - get_train_transforms()     : full augmentation pipeline for training
  - get_val_transforms()       : deterministic pipeline for validation
  - get_inference_transforms() : image-only pipeline for inference (no label)

Input format expected in data dictionaries
------------------------------------------
  "image" : list of 4 paths  [flair, t1, t1ce, t2]  (all .nii.gz)
  "label" : single path       seg mask               (all .nii.gz)

Label convention (BraTS 2020)
------------------------------
  0 → background
  1 → NCR/NET  (necrotic core / non-enhancing tumour)
  2 → ED       (peritumoral oedema)
  4 → ET       (enhancing tumour)

  We binarise to Whole Tumour (WT): labels {1, 2, 4} → 1, 0 → 0.
  This is the standard binary segmentation target for the vision agent.

Patch / memory budget
---------------------
  ROI_SIZE = (128, 128, 128) with batch_size=2 uses ~1.5–2 GB VRAM on the
  RTX 3050 (4 GB), leaving headroom for the LLM at inference time.
"""

from monai.transforms import (
    Compose,
    ConcatItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    SpatialPadd,
    DeleteItemsd,
)

# ---------------------------------------------------------------------------
# Shared constants — imported by dataset.py and train.py
# ---------------------------------------------------------------------------

#: Modality order used when stacking image channels.
#: dataset.py must build the image list in this exact order.
MODALITY_KEYS = ["flair", "t1", "t1ce", "t2"]

#: Spatial size of random patches extracted during training.
#: 128³ fits comfortably in ~1.5 GB VRAM with batch_size=2.
ROI_SIZE = (128, 128, 128)

#: Isotropic voxel spacing (mm) — BraTS scans are already 1 mm, but we
#: resample explicitly to guarantee consistency across all cases.
PIXDIM = (1.0, 1.0, 1.0)

#: Positive / negative patch sampling ratio for RandCropByPosNegLabeld.
#: 1 tumour-centred patch + 1 background patch per volume per step.
POS_SAMPLES = 1
NEG_SAMPLES = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _binarise_label(label):
    """Convert BraTS multi-class mask to binary Whole Tumour mask.

    Collapses labels 1 (NCR/NET), 2 (ED), 4 (ET) into a single class (1).
    Background (0) stays 0.

    Args:
        label: torch.Tensor — segmentation mask loaded by MONAI.

    Returns:
        torch.Tensor — binary mask, same dtype as input.
    """
    return (label > 0).float()


def _shared_load_and_preprocess(include_label: bool) -> list:
    """Return the transforms that are identical across all three pipelines.

    Covers: loading → channel stacking → label binarisation → spacing →
    orientation → intensity normalisation → foreground crop.

    Args:
        include_label: if True, label is included in all transform keys.

    Returns:
        list of MONAI transforms (not yet wrapped in Compose).
    """
    # Keys that every spatial transform touches
    spatial_keys = MODALITY_KEYS + (["label"] if include_label else [])
    image_keys   = MODALITY_KEYS           # only image modalities, no label

    transforms = [
        # ── 1. Load all NIfTI files ─────────────────────────────────────────
        # image key holds a list of 4 paths; MONAI loads each as a separate
        # entry in the dict under the modality key names set in MODALITY_KEYS.
        LoadImaged(
            keys=spatial_keys,
            image_only=True,
            ensure_channel_first=False,
        ),

        # ── 2. Add channel dim: [H,W,D] → [1,H,W,D] ────────────────────────
        EnsureChannelFirstd(keys=spatial_keys),

        # ── 3. Binarise segmentation label ──────────────────────────────────
        *(
            [Lambdad(keys=["label"], func=_binarise_label)]
            if include_label else []
        ),

        # ── 4. Concatenate 4 modalities → [4,H,W,D] ─────────────────────────
        ConcatItemsd(keys=MODALITY_KEYS, name="image", dim=0),

        # ── 5. Remove the now-redundant per-modality keys ───────────────────
        DeleteItemsd(keys=MODALITY_KEYS),

        # ── 6. Resample to isotropic 1 mm spacing ───────────────────────────
        Spacingd(
            keys=["image"] + (["label"] if include_label else []),
            pixdim=PIXDIM,
            mode=("bilinear", "nearest") if include_label else ("bilinear",),
        ),

        # ── 7. Standardise anatomical orientation to RAS ────────────────────
        Orientationd(
            keys=["image"] + (["label"] if include_label else []),
            axcodes="RAS",
        ),

        # ── 8. Z-score normalise each channel independently ─────────────────
        #   nonzero=True  → compute stats only on brain voxels (ignores skull-
        #                   stripped background zeros), which is the standard
        #                   practice for BraTS data.
        #   channel_wise=True → each of the 4 modalities gets its own μ/σ.
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True,
        ),

        # ── 9. Crop to tight bounding box around non-zero image region ───────
        #   Removes most of the zero-padded background, reducing memory and
        #   the chance that random patches land entirely in empty space.
        CropForegroundd(
            keys=["image"] + (["label"] if include_label else []),
            source_key="image",
            allow_smaller=True,
        ),

        # ── 10. Ensure float32 tensors ───────────────────────────────────────
        EnsureTyped(
            keys=["image"] + (["label"] if include_label else []),
            dtype="float32",
        ),
    ]

    return transforms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_train_transforms() -> Compose:
    """Build the full training transform pipeline.

    Deterministic steps (load → preprocess → normalise → crop) are followed
    by stochastic augmentations that run fresh each epoch.

    Expected data dict keys:
        flair, t1, t1ce, t2  → str paths to modality NIfTI files
        label                → str path to segmentation NIfTI file

    After this pipeline the dict will contain:
        image  → FloatTensor [4, 128, 128, 128]
        label  → FloatTensor [1, 128, 128, 128]  (binary WT mask)

    Returns:
        monai.transforms.Compose
    """
    base = _shared_load_and_preprocess(include_label=True)

    augmentations = [
        # ── Pad to minimum ROI_SIZE before random crop ───────────────────────
        #   Some BraTS cases are slightly smaller than 128 in one dimension
        #   after foreground cropping (e.g. 126×173×145).  SpatialPadd fills
        #   any deficient dimension with zeros so the crop always succeeds.
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=ROI_SIZE,
            mode="constant",        # zero-pad (background value)
        ),

        # ── Random 128³ patch ────────────────────────────────────────────────
        #   pos=1, neg=1 → half the patches are centred on tumour voxels,
        #   half on background.  This prevents the model from predicting all-
        #   background and solves class imbalance without oversampling.
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=ROI_SIZE,
            pos=POS_SAMPLES,
            neg=NEG_SAMPLES,
            num_samples=2,          # 2 patches extracted per volume per step
            image_key="image",
            image_threshold=0.0,
        ),

        # ── Spatial augmentations ────────────────────────────────────────────
        RandFlipd(
            keys=["image", "label"],
            prob=0.5,
            spatial_axis=0,         # left-right flip
        ),
        RandFlipd(
            keys=["image", "label"],
            prob=0.5,
            spatial_axis=1,         # anterior-posterior flip
        ),
        RandFlipd(
            keys=["image", "label"],
            prob=0.5,
            spatial_axis=2,         # superior-inferior flip
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.1,
            max_k=3,                # 0, 90, 180, or 270 degrees
        ),

        # ── Intensity augmentations ──────────────────────────────────────────
        #   Applied only to the image, never the label.
        #   Mild factors — aggressive intensity changes hurt convergence on MRI.
        RandScaleIntensityd(
            keys=["image"],
            factors=0.1,            # multiply by U[0.9, 1.1]
            prob=0.5,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.1,            # add U[-0.1, 0.1]
            prob=0.5,
        ),
    ]

    return Compose(base + augmentations)


def get_val_transforms() -> Compose:
    """Build the deterministic validation transform pipeline.

    No random cropping or augmentation — the full preprocessed volume is
    returned and sliding-window inference handles the patch extraction.

    Expected data dict keys: same as get_train_transforms().

    After this pipeline the dict will contain:
        image  → FloatTensor [4, H, W, D]   (full volume, no fixed size)
        label  → FloatTensor [1, H, W, D]   (binary WT mask)

    Returns:
        monai.transforms.Compose
    """
    base = _shared_load_and_preprocess(include_label=True)
    return Compose(base)


def get_inference_transforms() -> Compose:
    """Build the image-only inference transform pipeline.

    No label is required or expected.  Used by inference.py and agent.py
    when running a prediction on a single unseen scan.

    Expected data dict keys:
        flair, t1, t1ce, t2  → str paths to modality NIfTI files

    After this pipeline the dict will contain:
        image  → FloatTensor [4, H, W, D]   (full preprocessed volume)

    Returns:
        monai.transforms.Compose
    """
    base = _shared_load_and_preprocess(include_label=False)
    return Compose(base)
