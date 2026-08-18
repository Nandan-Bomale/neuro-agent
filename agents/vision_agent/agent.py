"""
agent.py
--------
VisionAgent — the public interface for the Vision Agent.

This is the only file the Orchestrator needs to import.  Everything else
(transforms, dataset, model, inference, gradcam) is an implementation detail
hidden behind this clean class.

Interface contract
------------------
  Input  : paths to 4 MRI modality NIfTI files + checkpoint path
  Output : VisionAgentResult dataclass containing:
              - segmentation_mask    (np.ndarray bool  [H, W, D])
              - confidence_score     (float 0–1)
              - probability_map      (np.ndarray float32 [H, W, D])
              - heatmap_volume       (np.ndarray float32 [H, W, D])
              - overlay_image        (np.ndarray uint8  [H, W, 3])
              - tumour_detected      (bool)
              - tumour_volume_voxels (int)
              - gradcam_slice        (int)
              - requires_review      (bool)  — True if confidence < threshold

The Orchestrator passes a VisionAgentResult to the Clinical History Agent,
Report Generation Agent, and Verification Agent.

Usage
-----
    from agents.vision_agent.agent import VisionAgent

    agent = VisionAgent(
        checkpoint_path="c:/Neuro Agent/models/vision/best_model.pth"
    )

    result = agent.run(
        flair_path="path/to/flair.nii",
        t1_path   ="path/to/t1.nii",
        t1ce_path ="path/to/t1ce.nii",
        t2_path   ="path/to/t2.nii",
    )

    print(f"Tumour detected  : {result.tumour_detected}")
    print(f"Confidence score : {result.confidence_score:.2f}")
    print(f"Requires review  : {result.requires_review}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from agents.vision_agent.gradcam import GradCAM
from agents.vision_agent.inference import predict
from agents.vision_agent.model import build_unet
from agents.vision_agent.transforms import get_inference_transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default path to the trained checkpoint.
DEFAULT_CHECKPOINT = "c:/Neuro Agent/models/vision/best_model.pth"

#: Confidence score below which the case is flagged for mandatory human review.
#: Also used by the Verification Agent as its primary signal.
REVIEW_THRESHOLD: float = 0.75


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VisionAgentResult:
    """Structured output from a single VisionAgent.run() call.

    All fields are populated by VisionAgent.run().  Pass this object directly
    to other agents in the pipeline — they read the fields they need.

    Attributes:
        segmentation_mask:    Binary whole-tumour mask [H, W, D].
        confidence_score:     Mean tumour-voxel sigmoid probability (0–1).
        probability_map:      Per-voxel sigmoid probabilities [H, W, D].
        heatmap_volume:       Grad-CAM saliency map [H, W, D], normalised 0–1.
        overlay_image:        2-D BGR overlay of the most salient MRI slice.
        tumour_detected:      True if any voxel was predicted as tumour.
        tumour_volume_voxels: Number of voxels predicted as tumour.
        gradcam_slice:        Axial slice index used for the overlay image.
        requires_review:      True if confidence_score < REVIEW_THRESHOLD.
        inference_time_s:     Wall-clock time for the full run() call (seconds).
    """
    segmentation_mask:    np.ndarray
    confidence_score:     float
    probability_map:      np.ndarray
    heatmap_volume:       np.ndarray
    overlay_image:        np.ndarray
    tumour_detected:      bool
    tumour_volume_voxels: int
    gradcam_slice:        int
    requires_review:      bool
    mri_slice_path:       Optional[str] = None
    inference_time_s:     float = field(default=0.0)

    def summary(self) -> str:
        """Return a human-readable one-paragraph summary of the result.

        This string is passed directly to the Report Generation Agent as
        structured context for the radiology report.

        Returns:
            str — plain-text summary suitable for LLM consumption.
        """
        status    = "detected" if self.tumour_detected else "not detected"
        review    = "FLAGGED FOR HUMAN REVIEW" if self.requires_review else "within acceptable confidence range"
        vol_cc    = self.tumour_volume_voxels / 1000.0  # voxels to ~cc (1mm isotropic voxels)

        return (
            f"Vision Agent Result:\n"
            f"  Tumour status   : {status}\n"
            f"  Tumour volume   : {self.tumour_volume_voxels} voxels (~{vol_cc:.1f} cc)\n"
            f"  Confidence score: {self.confidence_score:.3f} (threshold: {REVIEW_THRESHOLD})\n"
            f"  Review flag     : {review}\n"
            f"  Inference time  : {self.inference_time_s:.1f}s\n"
        )

    def to_dict(self) -> dict:
        """Serialise scalar fields to a plain dict (for JSON / LangGraph state).

        NumPy arrays are excluded — they are too large for the state graph and
        should be saved separately if needed.

        Returns:
            dict of serialisable scalar values.
        """
        return {
            "tumour_detected":      self.tumour_detected,
            "confidence_score":     round(self.confidence_score, 4),
            "tumour_volume_voxels": self.tumour_volume_voxels,
            "tumour_volume_cc":     round(self.tumour_volume_voxels / 1000.0, 2),
            "requires_review":      self.requires_review,
            "gradcam_slice":        self.gradcam_slice,
            "mri_slice_path":       self.mri_slice_path,
            "inference_time_s":     round(self.inference_time_s, 2),
        }


# ---------------------------------------------------------------------------
# VisionAgent class
# ---------------------------------------------------------------------------

class VisionAgent:
    """Vision Agent for brain MRI tumour segmentation.

    Wraps the full pipeline: preprocessing → U-Net inference → Grad-CAM →
    result packaging.  Loads the model once on construction; subsequent
    run() calls reuse the loaded weights.

    Args:
        checkpoint_path: Path to the trained U-Net checkpoint (.pth).
                         Trained by agents/vision_agent/train.py.
        device:          torch.device.  Auto-detected (CUDA if available).
        review_threshold: Confidence score below which requires_review=True.

    Raises:
        FileNotFoundError: if checkpoint_path does not exist.
    """

    def __init__(
        self,
        checkpoint_path: str   = DEFAULT_CHECKPOINT,
        device:          Optional[torch.device] = None,
        review_threshold: float = REVIEW_THRESHOLD,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device           = device
        self.checkpoint_path  = checkpoint_path
        self.review_threshold = review_threshold

        # Model is loaded once and kept in memory for repeated calls
        self._model:     Optional[torch.nn.Module] = None
        self._gradcam:   Optional[GradCAM]         = None
        self._transforms = get_inference_transforms()

        print(f"[VisionAgent] Initialised | device={device} | threshold={review_threshold}")

    # ── Lazy model loading ───────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """Load model and Grad-CAM hooks on first call (lazy init)."""
        if self._model is not None:
            return

        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"[VisionAgent] Checkpoint not found: {self.checkpoint_path}\n"
                "Train the model first: python -m agents.vision_agent.train"
            )

        self._model = build_unet(self.device)
        checkpoint  = torch.load(self.checkpoint_path, map_location=self.device)
        state       = checkpoint.get("model_state", checkpoint)
        self._model.load_state_dict(state)
        self._model.eval()

        self._gradcam = GradCAM(self._model)

        val_dice = checkpoint.get("val_dice", "unknown")
        epoch    = checkpoint.get("epoch",    "unknown")
        print(f"[VisionAgent] Model loaded | epoch={epoch} | val_dice={val_dice}")

    # ── Public API ───────────────────────────────────────────────────────────

    def run(
        self,
        mri_scan_path: str,
        target_slice:  Optional[int] = None,
        modality_idx:  int = 0,
    ) -> VisionAgentResult:
        """Run the full Vision Agent pipeline on one patient's MRI scan.

        Steps:
          1. Preprocess 4 modality NIfTI files (spacing, orientation, normalise)
          2. Sliding-window U-Net inference → segmentation mask + confidence
          3. Grad-CAM → heatmap volume + 2-D overlay image
          4. Package everything into a VisionAgentResult

        Args:
            mri_scan_path: Path to the NIfTI MRI file (3D or 4D).
            target_slice:  Axial slice index for the Grad-CAM overlay.
                           If None, the most salient slice is chosen automatically.
            modality_idx:  MRI channel used as the overlay background (0=FLAIR).

        Returns:
            VisionAgentResult — fully populated result object.
        """
        t_start = time.time()

        print("[VisionAgent] Starting inference pipeline...")
        self._ensure_model_loaded()

        # ── Step 1 + 2: Inference (predict handles preprocessing internally) ─
        inference_result = predict(
            mri_scan_path   = mri_scan_path,
            checkpoint_path = self.checkpoint_path,
            device          = self.device,
        )

        # ── Step 3: Grad-CAM ─────────────────────────────────────────────────
        print("[VisionAgent] Generating Grad-CAM heatmap...")

        # Re-run preprocessing to get the tensor
        from agents.vision_agent.transforms import get_single_inference_transforms
        data_dict = {
            "image": mri_scan_path,
        }
        transforms = get_single_inference_transforms()
        processed     = transforms(data_dict)
        image_tensor  = processed["image"].unsqueeze(0).to(self.device)

        gradcam_result = self._gradcam.generate(
            image_tensor = image_tensor,
            target_slice = target_slice,
            modality_idx = modality_idx,
        )

        # ── Extract and Save the 2D Slice for Tumor Classification ───────────
        target_s = gradcam_result["target_slice"]
        # Extract the raw MRI slice from the tensor: shape [H, W]
        # Modality idx is used (typically 0 or 2 for T1ce)
        raw_slice = image_tensor[0, modality_idx, :, :, target_s].detach().cpu().numpy()
        
        # Normalize to 0-255 uint8
        raw_slice = raw_slice - raw_slice.min()
        if raw_slice.max() > 0:
            raw_slice = raw_slice / raw_slice.max()
        raw_slice = (raw_slice * 255).astype(np.uint8)
        
        # Save as JPG
        import os
        from PIL import Image
        slice_dir = Path("data/interim/extracted_slices")
        slice_dir.mkdir(parents=True, exist_ok=True)
        slice_name = f"extracted_slice_{int(time.time())}.jpg"
        slice_path = slice_dir / slice_name
        
        # Convert to RGB (3-channel) as Tumor Classification expects RGB images
        img = Image.fromarray(raw_slice).convert("RGB")
        img.save(slice_path)
        print(f"[VisionAgent] Saved 2D slice for classification: {slice_path}")

        # ── Step 4: Package result ───────────────────────────────────────────
        confidence   = inference_result["confidence_score"]
        elapsed      = time.time() - t_start

        result = VisionAgentResult(
            segmentation_mask    = inference_result["segmentation_mask"],
            confidence_score     = confidence,
            probability_map      = inference_result["probability_map"],
            heatmap_volume       = gradcam_result["heatmap_volume"],
            overlay_image        = gradcam_result["overlay_image"],
            tumour_detected      = inference_result["tumour_detected"],
            tumour_volume_voxels = inference_result["tumour_volume_voxels"],
            gradcam_slice        = gradcam_result["target_slice"],
            requires_review      = confidence < self.review_threshold,
            mri_slice_path       = str(slice_path),
            inference_time_s     = elapsed,
        )

        print("[VisionAgent] Done.")
        print(result.summary())

        return result

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"VisionAgent("
            f"device={self.device}, "
            f"checkpoint='{Path(self.checkpoint_path).name}', "
            f"model_loaded={loaded}, "
            f"review_threshold={self.review_threshold})"
        )
