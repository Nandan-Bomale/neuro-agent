"""
inference.py
------------
Single-scan inference pipeline for the Vision Agent.

Takes paths to one patient's 4 MRI modality files, loads a trained U-Net
checkpoint, and returns:
  - segmentation_mask : np.ndarray [H, W, D]  — binary whole-tumour mask
  - confidence_score  : float  [0.0, 1.0]     — mean sigmoid probability
                                                 over predicted tumour voxels
  - probability_map   : np.ndarray [H, W, D]  — raw sigmoid probabilities
                                                 (used by gradcam.py for overlay)

The output is in the same spatial space as the input scan (after resampling
to 1 mm isotropic and RAS orientation).  Sliding-window inference with 50%
overlap is used so the full volume is covered regardless of size.

Usage
-----
    from agents.vision_agent.inference import predict

    result = predict(
        flair_path="path/to/flair.nii",
        t1_path   ="path/to/t1.nii",
        t1ce_path ="path/to/t1ce.nii",
        t2_path   ="path/to/t2.nii",
        checkpoint ="c:/Neuro Agent/models/vision/best_model.pth",
    )

    mask       = result["segmentation_mask"]   # np.ndarray, binary
    confidence = result["confidence_score"]    # float
    prob_map   = result["probability_map"]     # np.ndarray, float32
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete

from agents.vision_agent.model import build_unet
from agents.vision_agent.transforms import (
    ROI_SIZE,
    get_inference_transforms,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default threshold for binarising the sigmoid probability map.
DEFAULT_THRESHOLD: float = 0.5

#: Sliding-window inference overlap fraction.
SW_OVERLAP: float = 0.5

#: Sliding-window stitching mode.
SW_MODE: str = "gaussian"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load U-Net weights from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pth checkpoint saved by train.py.
        device:          torch.device to load the model onto.

    Returns:
        nn.Module — U-Net in eval mode with weights loaded.

    Raises:
        FileNotFoundError: if the checkpoint file does not exist.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first using agents/vision_agent/train.py"
        )

    model = build_unet(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict and the dict saved by train.py
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()

    val_dice = checkpoint.get("val_dice", "unknown")
    epoch    = checkpoint.get("epoch",    "unknown")
    print(f"[inference] Loaded checkpoint: epoch={epoch}, val_dice={val_dice}")

    return model


def _build_data_dict(
    flair_path: str,
    t1_path:    str,
    t1ce_path:  str,
    t2_path:    str,
) -> dict:
    """Build the MONAI-compatible data dictionary for one patient.

    Keys match MODALITY_KEYS in transforms.py: flair, t1, t1ce, t2.

    Args:
        flair_path: Path to FLAIR modality .nii / .nii.gz file.
        t1_path:    Path to T1    modality .nii / .nii.gz file.
        t1ce_path:  Path to T1ce  modality .nii / .nii.gz file.
        t2_path:    Path to T2    modality .nii / .nii.gz file.

    Returns:
        dict ready to pass into the inference transform pipeline.
    """
    for path in (flair_path, t1_path, t1ce_path, t2_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"MRI file not found: {path}")

    return {
        "flair": flair_path,
        "t1":    t1_path,
        "t1ce":  t1ce_path,
        "t2":    t2_path,
    }


def _compute_confidence(prob_map: np.ndarray, mask: np.ndarray) -> float:
    """Compute a confidence score from the probability map.

    Confidence = mean sigmoid probability over voxels predicted as tumour.
    If no tumour is predicted, falls back to the mean over all voxels (which
    will be low, correctly reflecting low confidence in a "no tumour" call).

    This is a useful signal for the Verification Agent — low confidence
    (e.g. < 0.75) should trigger human review.

    Args:
        prob_map: float32 array [H, W, D] — sigmoid probabilities.
        mask:     binary  array [H, W, D] — thresholded prediction.

    Returns:
        float in [0, 1].
    """
    tumour_voxels = prob_map[mask.astype(bool)]
    if tumour_voxels.size == 0:
        # No tumour predicted — return the mean background probability
        # (will typically be very low, e.g. 0.05–0.15)
        return float(prob_map.mean())
    return float(tumour_voxels.mean())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(
    flair_path:      str,
    t1_path:         str,
    t1ce_path:       str,
    t2_path:         str,
    checkpoint_path: str = "c:/Neuro Agent/models/vision/best_model.pth",
    threshold:       float = DEFAULT_THRESHOLD,
    device:          torch.device | None = None,
) -> Dict:
    """Run inference on a single patient's MRI scan.

    This is the function called by VisionAgent.run() in agent.py.

    Args:
        flair_path:      Path to FLAIR modality NIfTI file.
        t1_path:         Path to T1    modality NIfTI file.
        t1ce_path:       Path to T1ce  modality NIfTI file.
        t2_path:         Path to T2    modality NIfTI file.
        checkpoint_path: Path to trained U-Net checkpoint (.pth).
        threshold:       Sigmoid probability threshold for binarisation.
        device:          torch.device. Auto-detected if None.

    Returns:
        dict with keys:
            segmentation_mask (np.ndarray, bool, [H,W,D])
                Binary whole-tumour mask in the preprocessed scan space.

            confidence_score (float, 0–1)
                Mean tumour-voxel sigmoid probability.
                < 0.5  → model is uncertain, review recommended
                > 0.75 → model is confident

            probability_map (np.ndarray, float32, [H,W,D])
                Per-voxel sigmoid probabilities (used by Grad-CAM overlay).

            tumour_detected (bool)
                True if any voxel was predicted as tumour.

            tumour_volume_voxels (int)
                Number of voxels predicted as tumour.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[inference] Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    model = _load_model(checkpoint_path, device)

    # ── Load and preprocess scan ────────────────────────────────────────────
    print("[inference] Preprocessing scan...")
    data_dict  = _build_data_dict(flair_path, t1_path, t1ce_path, t2_path)
    transforms = get_inference_transforms()
    processed  = transforms(data_dict)

    # Add batch dimension: [4, H, W, D] → [1, 4, H, W, D]
    image_tensor = processed["image"].unsqueeze(0).to(device)

    print(f"[inference] Input tensor shape: {list(image_tensor.shape)}")

    # ── Sliding-window inference ────────────────────────────────────────────
    print("[inference] Running sliding-window inference...")
    sigmoid_fn = Activations(sigmoid=True)
    binarise   = AsDiscrete(threshold=threshold)

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image_tensor,
            roi_size=ROI_SIZE,
            sw_batch_size=2,
            predictor=model,
            overlap=SW_OVERLAP,
            mode=SW_MODE,
        )
        # logits: [1, 1, H, W, D]
        probs  = sigmoid_fn(logits)         # [1, 1, H, W, D]
        binary = binarise(probs)            # [1, 1, H, W, D]

    # ── Convert to numpy and squeeze to [H, W, D] ──────────────────────────
    prob_map = probs[0, 0].cpu().numpy()    # float32 [H, W, D]
    mask     = binary[0, 0].cpu().numpy().astype(bool)   # bool [H, W, D]

    # ── Compute outputs ─────────────────────────────────────────────────────
    confidence        = _compute_confidence(prob_map, mask)
    tumour_detected   = bool(mask.any())
    tumour_volume     = int(mask.sum())

    print(f"[inference] Tumour detected    : {tumour_detected}")
    print(f"[inference] Tumour volume      : {tumour_volume} voxels")
    print(f"[inference] Confidence score   : {confidence:.4f}")

    return {
        "segmentation_mask":    mask,
        "confidence_score":     confidence,
        "probability_map":      prob_map,
        "tumour_detected":      tumour_detected,
        "tumour_volume_voxels": tumour_volume,
    }
