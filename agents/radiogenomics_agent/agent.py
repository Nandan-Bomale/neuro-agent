"""
agent.py
--------
RadiogenomicsAgent — Predicts IDH mutation and MGMT methylation status
from a 3D MRI scan using the trained DenseNet-121 3D CNN.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

AGENT_NAME   = "radiogenomics_agent"
DEFAULT_CKPT = "models/radiogenomics/radiogenomics_best.pth"

# Match the ROI size used during training
ROI_SIZE = (128, 128, 128)


class RadiogenomicsAgent:
    """3D Radiogenomics CNN Agent — IDH + MGMT prediction from MRI."""

    def __init__(
        self,
        ckpt_path: str = DEFAULT_CKPT,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ckpt_path = ckpt_path
        self.device    = device
        self.model     = None
        logger.info(f"[{AGENT_NAME}] Initialized | device={device} | ckpt={ckpt_path}")

    # ── Lazy model loading ──────────────────────────────────────────────────

    def _ensure_model(self):
        if self.model is not None:
            return

        if not Path(self.ckpt_path).exists():
            logger.warning(
                f"[{AGENT_NAME}] Checkpoint not found at {self.ckpt_path}. "
                "Running in MOCK mode — run 'git pull origin dev' on DGX to fetch weights."
            )
            self.model = "mock"
            return

        try:
            from agents.radiogenomics_agent.model import RadiogenomicsNet
            net = RadiogenomicsNet(num_classes=2)
            state = torch.load(self.ckpt_path, map_location=self.device)
            net.load_state_dict(state)
            net.to(self.device).eval()
            self.model = net
            logger.info(f"[{AGENT_NAME}] Model loaded from {self.ckpt_path}")
        except Exception as e:
            logger.error(f"[{AGENT_NAME}] Failed to load model: {e}. Falling back to mock.")
            self.model = "mock"

    # ── Preprocessing ────────────────────────────────────────────────────────

    def _preprocess(self, scan_path: str) -> torch.Tensor:
        """Load a single 3D NIfTI, resize to ROI_SIZE, return (1,4,D,H,W) tensor.
        
        Since agent.py only receives one path (the 3D MRI), we use it for all
        4 channels (FLAIR/T1/T1ce/T2). The model will still extract spatial 
        radiogenomic features regardless of channel duplication.
        """
        try:
            import nibabel as nib
            from monai.transforms import (
                LoadImage, EnsureChannelFirst, NormalizeIntensity,
                Resize, ToTensor
            )

            loader = LoadImage(image_only=True)
            vol = loader(scan_path)             # (H, W, D) or (C, H, W, D)
            if vol.ndim == 3:
                vol = vol.unsqueeze(0)           # → (1, H, W, D)

            # Resize spatial dims to ROI_SIZE
            resize = Resize(spatial_size=ROI_SIZE, mode="trilinear")
            vol = resize(vol)                   # (1, H, W, D)

            # Normalize
            norm = NormalizeIntensity(nonzero=True)
            vol = norm(vol)

            # Replicate single channel → 4 channels (FLAIR=T1=T1ce=T2)
            vol4 = vol.repeat(4, 1, 1, 1)       # (4, H, W, D)

            return vol4.unsqueeze(0).to(self.device)  # (1, 4, H, W, D)

        except Exception as e:
            logger.error(f"[{AGENT_NAME}] Preprocessing failed: {e}")
            raise

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the Radiogenomics Agent.
        
        Reads:  state["mri_scan_path"]
        Writes: state["radiogenomics_findings"]
        """
        t_start = time.perf_counter()
        logger.info(f"[{AGENT_NAME}] run() invoked.")

        mri_scan_path = state.get("mri_scan_path")
        if not mri_scan_path:
            logger.error(f"[{AGENT_NAME}] Missing 'mri_scan_path' in state.")
            return {"radiogenomics_findings": {}}

        self._ensure_model()

        # ── MOCK mode (no checkpoint) ────────────────────────────────────────
        if self.model == "mock":
            logger.info(f"[{AGENT_NAME}] Running in MOCK mode.")
            elapsed = round(time.perf_counter() - t_start, 3)
            return {
                "radiogenomics_findings": {
                    "idh_mutation_status":   "mutant",
                    "mgmt_methylation_status": "methylated",
                    "idh_confidence":        0.89,
                    "mgmt_confidence":       0.87,
                    "inference_time_sec":    elapsed,
                    "source":                "mock",
                }
            }

        # ── REAL inference ───────────────────────────────────────────────────
        try:
            tensor = self._preprocess(mri_scan_path)  # (1, 4, D, H, W)

            with torch.no_grad():
                logits = self.model(tensor)            # (1, 2) — IDH + MGMT

            probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # shape (2,)

            idh_prob  = float(probs[0])
            mgmt_prob = float(probs[1])

            idh_status  = "mutant"      if idh_prob  > 0.5 else "wildtype"
            mgmt_status = "methylated"  if mgmt_prob > 0.5 else "unmethylated"

            elapsed = round(time.perf_counter() - t_start, 3)
            logger.info(
                f"[{AGENT_NAME}] Done in {elapsed}s | IDH={idh_status}({idh_prob:.2f}) "
                f"| MGMT={mgmt_status}({mgmt_prob:.2f})"
            )

            return {
                "radiogenomics_findings": {
                    "idh_mutation_status":    idh_status,
                    "mgmt_methylation_status": mgmt_status,
                    "idh_confidence":         round(idh_prob, 4),
                    "mgmt_confidence":        round(mgmt_prob, 4),
                    "inference_time_sec":     elapsed,
                    "source":                 "trained_model",
                }
            }

        except Exception as e:
            logger.error(f"[{AGENT_NAME}] Inference failed: {e}. Returning mock output.")
            return {
                "radiogenomics_findings": {
                    "idh_mutation_status":    "wildtype",
                    "mgmt_methylation_status": "unmethylated",
                    "idh_confidence":         0.0,
                    "mgmt_confidence":        0.0,
                    "inference_time_sec":     round(time.perf_counter() - t_start, 3),
                    "source":                 "error_fallback",
                }
            }
