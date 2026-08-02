"""
gradcam.py
----------
Grad-CAM heatmap generation for the Vision Agent U-Net.

Grad-CAM (Gradient-weighted Class Activation Mapping) shows *which regions
of the input scan had the most influence* on the model's tumour prediction.
The heatmap is overlaid on a 2-D MRI slice for visual inspection by the
radiologist / demo viewer.

How it works
------------
  1. Register a forward hook on the target layer (last encoder conv) to
     capture its activation map during the forward pass.
  2. Register a backward hook on the same layer to capture gradients w.r.t.
     the segmentation loss.
  3. Pool the gradients spatially → per-channel importance weights.
  4. Weight-sum the activations → Grad-CAM volume [H, W, D].
  5. ReLU + normalise → [0, 1] saliency volume.
  6. Interpolate to input resolution and overlay on the chosen MRI slice.

Target layer
------------
  The last convolutional block of the U-Net encoder (before the bottleneck)
  is used.  At this depth the feature maps capture high-level tumour
  semantics rather than low-level edges.  MONAI's UNet architecture names
  this as `model.model[0]` (the outermost SequentialResBlock).

Output
------
  - heatmap_volume : np.ndarray float32 [H, W, D] — normalised [0,1]
  - overlay_image  : np.ndarray uint8   [H, W, 3]  — BGR OpenCV image
                     of the chosen axial slice with the heatmap overlaid

Usage
-----
    from agents.vision_agent.gradcam import GradCAM

    gcam = GradCAM(model)
    result = gcam.generate(
        image_tensor  = processed["image"].unsqueeze(0).to(device),
        target_slice  = 80,        # axial slice index to visualise
        modality_idx  = 0,         # which channel to use as background (0=FLAIR)
    )
    heatmap = result["heatmap_volume"]
    overlay = result["overlay_image"]
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from monai.transforms import Activations


# ---------------------------------------------------------------------------
# GradCAM class
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM heatmap generator for a 3-D segmentation U-Net.

    Args:
        model:        Trained U-Net nn.Module (output of build_unet()).
        target_layer: The nn.Module layer to hook into.  If None, the last
                      encoder block of the MONAI UNet is used automatically.

    Example::

        model = build_unet()
        gcam  = GradCAM(model)
        result = gcam.generate(image_tensor, target_slice=80)
    """

    def __init__(
        self,
        model:        nn.Module,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        self.model  = model
        self.device = next(model.parameters()).device

        # Auto-select the last encoder conv block if none provided
        if target_layer is None:
            target_layer = self._get_default_target_layer(model)
        self.target_layer = target_layer

        # Storage for hook outputs
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = self.target_layer.register_forward_hook(
            self._forward_hook
        )
        self._bwd_hook = self.target_layer.register_full_backward_hook(
            self._backward_hook
        )

    # ── Hooks ────────────────────────────────────────────────────────────────

    def _forward_hook(
        self,
        module: nn.Module,
        input:  tuple,
        output: torch.Tensor,
    ) -> None:
        """Capture activations from the target layer during forward pass."""
        self._activations = output.detach().clone()

    def _backward_hook(
        self,
        module:       nn.Module,
        grad_input:   tuple,
        grad_output:  tuple,
    ) -> None:
        """Capture gradients flowing into the target layer during backward."""
        self._gradients = grad_output[0].detach().clone()

    # ── Layer selection ──────────────────────────────────────────────────────

    @staticmethod
    def _get_default_target_layer(model: nn.Module) -> nn.Module:
        """Return the last encoder block of a MONAI UNet.

        MONAI's UNet wraps everything in a top-level Sequential.  The last
        *encoder* block sits at index -2 of model.model (index -1 is the
        classification head).  We recurse into it to get the innermost conv.

        If the structure has changed (e.g. different MONAI version), we fall
        back to the last Conv3d layer in the network.

        Args:
            model: MONAI UNet.

        Returns:
            nn.Module — the target layer for Grad-CAM hooks.
        """
        try:
            # MONAI UNet: model.model is a Sequential of StackedConv blocks
            # The encoder blocks are the first half; we want the deepest one.
            encoder_blocks = list(model.model.children())
            # Last meaningful encoder block (before the decoder path)
            target = encoder_blocks[len(encoder_blocks) // 2]
            return target
        except (AttributeError, IndexError):
            # Fallback: find the last Conv3d in the whole network
            last_conv = None
            for module in model.modules():
                if isinstance(module, nn.Conv3d):
                    last_conv = module
            if last_conv is None:
                raise RuntimeError(
                    "Could not find a Conv3d layer in the model for Grad-CAM."
                )
            return last_conv

    # ── Core Grad-CAM computation ────────────────────────────────────────────

    def _compute_gradcam_volume(
        self,
        image_tensor: torch.Tensor,
    ) -> np.ndarray:
        """Run one forward+backward pass and compute the Grad-CAM volume.

        Args:
            image_tensor: float32 tensor [1, 4, H, W, D] on the model's device.

        Returns:
            np.ndarray float32 [H, W, D] — normalised Grad-CAM saliency map.
        """
        self.model.eval()
        self._activations = None
        self._gradients   = None

        # ── Pad to multiple of 16 ────────────────────────────────────────────
        # The U-Net has 4 stride-2 downsampling stages (2^4 = 16).  When the
        # full volume is passed directly (not via sliding window), spatial dims
        # must be divisible by 16, otherwise skip-connection concatenations
        # fail with a size mismatch.  We pad, compute, then interpolate back.
        STRIDE_PROD = 16
        orig_spatial = image_tensor.shape[2:]          # (H, W, D)
        H, W, D = orig_spatial

        pH = (STRIDE_PROD - H % STRIDE_PROD) % STRIDE_PROD
        pW = (STRIDE_PROD - W % STRIDE_PROD) % STRIDE_PROD
        pD = (STRIDE_PROD - D % STRIDE_PROD) % STRIDE_PROD

        # F.pad order (innermost dim first): (D_start,D_end, W_start,W_end, H_start,H_end)
        padded = torch.nn.functional.pad(image_tensor, (0, pD, 0, pW, 0, pH))

        # ── Forward ─────────────────────────────────────────────────────────
        padded.requires_grad_(False)
        logits = self.model(padded)                    # [1, 1, H', W', D']

        # ── Target signal ────────────────────────────────────────────────────
        sigmoid_fn = Activations(sigmoid=True)
        probs  = sigmoid_fn(logits)
        target = probs.mean()

        # ── Backward ─────────────────────────────────────────────────────────
        self.model.zero_grad()
        target.backward()

        # ── Grad-CAM formula ─────────────────────────────────────────────────
        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire.  Check that target_layer is "
                "actually used during the forward pass."
            )

        weights    = self._gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam_volume = (weights * self._activations).sum(dim=1, keepdim=True)
        cam_volume = torch.relu(cam_volume)

        # Resize back to ORIGINAL (unpadded) spatial dims
        cam_volume = torch.nn.functional.interpolate(
            cam_volume,
            size=orig_spatial,           # (H, W, D) — not the padded size
            mode="trilinear",
            align_corners=False,
        )

        # Normalise to [0, 1]
        cam_np = cam_volume[0, 0].cpu().numpy()        # [H, W, D]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np.astype(np.float32)


    # ── Overlay rendering ────────────────────────────────────────────────────

    @staticmethod
    def _render_overlay(
        mri_slice:   np.ndarray,
        cam_slice:   np.ndarray,
        alpha:       float = 0.45,
    ) -> np.ndarray:
        """Blend a 2-D Grad-CAM heatmap onto a greyscale MRI slice.

        Args:
            mri_slice: float array [H, W] — one 2-D slice of the MRI volume,
                       normalised to [0, 1].
            cam_slice: float array [H, W] — corresponding Grad-CAM slice,
                       normalised to [0, 1].
            alpha:     Heatmap opacity (0 = invisible, 1 = fully opaque).

        Returns:
            np.ndarray uint8 [H, W, 3] — BGR OpenCV image.
        """
        # Normalise MRI slice to [0, 255] uint8 greyscale
        mri_norm = (mri_slice - mri_slice.min())
        if mri_norm.max() > 1e-8:
            mri_norm = mri_norm / mri_norm.max()
        mri_uint8 = (mri_norm * 255).astype(np.uint8)
        mri_bgr   = cv2.cvtColor(mri_uint8, cv2.COLOR_GRAY2BGR)

        # Colourmap the Grad-CAM slice (JET: blue=low, red=high saliency)
        cam_uint8 = (cam_slice * 255).astype(np.uint8)
        cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        # Weighted blend
        overlay = cv2.addWeighted(mri_bgr, 1 - alpha, cam_color, alpha, 0)
        return overlay

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        image_tensor:  torch.Tensor,
        target_slice:  Optional[int] = None,
        modality_idx:  int = 0,
        alpha:         float = 0.45,
    ) -> Dict:
        """Generate Grad-CAM heatmap and 2-D overlay for one patient volume.

        Args:
            image_tensor: float32 tensor [1, 4, H, W, D] — preprocessed scan
                          as returned by get_inference_transforms().
            target_slice: Axial slice index (z-axis) to visualise.  If None,
                          uses the slice with the highest mean Grad-CAM value
                          (automatically finds the most salient slice).
            modality_idx: Which MRI channel to use as the greyscale background
                          (0=FLAIR, 1=T1, 2=T1ce, 3=T2).  FLAIR is best for
                          whole-tumour visualisation.
            alpha:        Heatmap opacity on the overlay (0–1).

        Returns:
            dict with keys:
                heatmap_volume (np.ndarray, float32, [H,W,D])
                    Full 3-D normalised Grad-CAM saliency map.

                overlay_image (np.ndarray, uint8, [H,W,3])
                    2-D BGR image of the chosen axial slice + heatmap overlay.

                target_slice (int)
                    The axial slice index that was visualised.
        """
        # Compute Grad-CAM volume
        heatmap_volume = self._compute_gradcam_volume(image_tensor)

        # Select the axial slice to visualise
        if target_slice is None:
            # Search only the middle third of the volume — edge slices are
            # typically all-background and produce near-zero Grad-CAM values,
            # causing argmax to pick a meaningless edge slice.
            D = heatmap_volume.shape[2]
            z_start = D // 3
            z_end   = 2 * D // 3
            mid_region = heatmap_volume[:, :, z_start:z_end]     # [H, W, D/3]
            best_local  = int(np.argmax(mid_region.max(axis=(0, 1))))
            target_slice = z_start + best_local

        # Extract 2-D slices
        mri_slice = image_tensor[0, modality_idx, :, :, target_slice].cpu().numpy()
        cam_slice = heatmap_volume[:, :, target_slice]

        # Render overlay
        overlay = self._render_overlay(mri_slice, cam_slice, alpha=alpha)

        return {
            "heatmap_volume": heatmap_volume,
            "overlay_image":  overlay,
            "target_slice":   target_slice,
        }

    def remove_hooks(self) -> None:
        """Remove the registered forward and backward hooks.

        Call this when you are done with Grad-CAM to avoid memory leaks,
        especially if the model will continue to be used for inference.
        """
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self) -> None:
        """Clean up hooks on garbage collection."""
        try:
            self.remove_hooks()
        except Exception:
            pass
