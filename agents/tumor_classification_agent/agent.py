"""
agent.py
--------
TumorClassificationAgent — the single public interface the Orchestrator imports.

Pipeline
--------
    Stage 1 (always):
        TumorTypeEnsemble predicts one of:
            glioma | meningioma | notumor | pituitary

    Stage 2 (only when Stage 1 → glioma):
        TumorGradeClassifier predicts one of:
            grade_II | grade_III | grade_IV

Input  (state dict)
-------------------
    Required — one of:
        "mri_slice_path"   (str | Path)   path to any PIL-readable 2-D MRI slice
        "mri_slice_array"  (np.ndarray)   H×W, H×W×1, or H×W×3 array

    Optional:
        "vision_findings"  (dict)          VisionAgent output, used for context logging

Output (dict)
-------------
    {
        "agent_name": "tumor_classification_agent",
        "success":    True,
        "confidence": 0.94,
        "output": {
            "tumor_type":          "glioma",
            "tumor_grade":         "grade_IV",     # None if not glioma
            "type_probabilities":  {"glioma": 0.94, "meningioma": 0.03,
                                    "notumor": 0.01, "pituitary": 0.02},
            "grade_probabilities": {"grade_II": 0.05, "grade_III": 0.18,
                                    "grade_IV": 0.77},  # None if not glioma
            "clinical_urgency":    "urgent",        # urgent | monitor | routine
            "tta_used":            True,
        },
        "error": None
    }

Clinical urgency
----------------
    urgent  : glioma grade_IV  OR  glioma grade_III
    monitor : glioma grade_II  OR  meningioma  OR  glioma (grade unknown)
    routine : notumor  OR  pituitary

Usage
-----
    from agents.tumor_classification_agent.agent import TumorClassificationAgent

    agent = TumorClassificationAgent(
        type_ckpt_path  = "models/tumor_classifier/type_ensemble_best.pth",
        grade_ckpt_path = "models/tumor_classifier/grade_classifier_best.pth",
    )

    result = agent.run({
        "mri_slice_path": "data/sample_slice.jpg",
        "vision_findings": {...},   # optional
    })
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from agents.tumor_classification_agent.inference import TumorPredictor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_NAME = "tumor_classification_agent"

DEFAULT_TYPE_CKPT  = "models/tumor_classifier/type_ensemble_best.pth"
DEFAULT_GRADE_CKPT = "models/tumor_classifier/grade_classifier_best.pth"


# ---------------------------------------------------------------------------
# Clinical urgency mapping
# ---------------------------------------------------------------------------

def _clinical_urgency(tumor_type: str, tumor_grade: Optional[str]) -> str:
    """Map (tumor_type, tumor_grade) → clinical urgency string.

    Rules:
        urgent  → glioma grade_IV or glioma grade_III
        monitor → glioma grade_II | meningioma | glioma (grade unavailable)
        routine → notumor | pituitary | any unknown type

    Args:
        tumor_type:  Stage-1 prediction (e.g., "glioma").
        tumor_grade: Stage-2 prediction or None.

    Returns:
        One of "urgent" | "monitor" | "routine".
    """
    if tumor_type == "glioma":
        if tumor_grade in ("grade_III", "grade_IV"):
            return "urgent"
        # grade_II, or grade unavailable
        return "monitor"
    if tumor_type == "meningioma":
        return "monitor"
    # notumor, pituitary, or unrecognised
    return "routine"


# ---------------------------------------------------------------------------
# TumorClassificationAgent
# ---------------------------------------------------------------------------

class TumorClassificationAgent:
    """Brain tumor type + glioma grade classification agent.

    Wraps a two-stage TumorPredictor behind the standard Orchestrator
    ``.run(state)`` interface.  The TumorPredictor is instantiated lazily
    on the first call to ``.run()`` so that import-time is fast even when
    checkpoints are not yet present.

    Args:
        type_ckpt_path:   Path to the Stage-1 ensemble checkpoint (.pth).
        grade_ckpt_path:  Path to the Stage-2 grade classifier checkpoint (.pth).
        device:           Compute device.  Auto-detected (CUDA if available).
        tta:              Enable Test-Time Augmentation (default True).
        tta_n:            Number of TTA augmented views (default 10).
    """

    def __init__(
        self,
        type_ckpt_path:  str                    = DEFAULT_TYPE_CKPT,
        grade_ckpt_path: str                    = DEFAULT_GRADE_CKPT,
        device:          Optional[torch.device] = None,
        tta:             bool                   = True,
        tta_n:           int                    = 10,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device          = device
        self.type_ckpt_path  = type_ckpt_path
        self.grade_ckpt_path = grade_ckpt_path
        self.tta             = tta
        self.tta_n           = tta_n

        self._predictor: Optional[TumorPredictor] = None

        print(
            f"[{AGENT_NAME}] Initialised | device={device} | "
            f"TTA={tta} (n={tta_n})"
        )

    # ── Lazy predictor ───────────────────────────────────────────────────────

    def _ensure_predictor(self) -> None:
        """Build TumorPredictor on first run() call."""
        if self._predictor is not None:
            return
        self._predictor = TumorPredictor(
            type_ckpt  = self.type_ckpt_path,
            grade_ckpt = self.grade_ckpt_path,
            device     = self.device,
            tta        = self.tta,
            tta_n      = self.tta_n,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the Tumor Classification Agent on one MRI slice.

        Args:
            state: Dictionary containing input and optional context.

                Required (one of):
                    "mri_slice_path"   (str | Path)  — path to 2-D MRI slice image
                    "mri_slice_array"  (np.ndarray)  — H×W or H×W×C numpy array

                Optional:
                    "vision_findings"  (dict)  — VisionAgent output for logging

        Returns:
            Standard Orchestrator result dict (see module docstring).
        """
        t_start = time.perf_counter()
        print(f"\n[{AGENT_NAME}] run() invoked.")

        # ── Extract MRI source ────────────────────────────────────────────────
        mri_source: Optional[Union[str, np.ndarray]] = None

        if state.get("mri_slice_path") is not None:
            mri_source = str(state["mri_slice_path"])
            print(f"[{AGENT_NAME}] Input: path={mri_source}")

        elif state.get("mri_slice_array") is not None:
            mri_source = state["mri_slice_array"]
            shape = mri_source.shape if hasattr(mri_source, "shape") else "?"
            print(f"[{AGENT_NAME}] Input: numpy array shape={shape}")

        else:
            return self._fail(
                "state must contain 'mri_slice_path' (str) or "
                "'mri_slice_array' (np.ndarray)."
            )

        # ── Log VisionAgent context ───────────────────────────────────────────
        vf = state.get("vision_findings")
        if vf:
            print(
                f"[{AGENT_NAME}] VisionAgent context: "
                f"tumour_detected={vf.get('tumour_detected')} | "
                f"confidence={vf.get('confidence_score')}"
            )

        # ── Run two-stage inference ───────────────────────────────────────────
        try:
            self._ensure_predictor()
            pred = self._predictor.predict(mri_source)
        except FileNotFoundError as exc:
            return self._fail(str(exc))
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[{AGENT_NAME}] Unexpected error:\n{tb}")
            return self._fail(str(exc))

        # ── Derive clinical urgency ───────────────────────────────────────────
        urgency = _clinical_urgency(pred["tumor_type"], pred["tumor_grade"])
        elapsed = round(time.perf_counter() - t_start, 3)

        print(
            f"[{AGENT_NAME}] Done in {elapsed}s | "
            f"type={pred['tumor_type']} | grade={pred['tumor_grade']} | "
            f"urgency={urgency} | confidence={pred['confidence']:.4f}"
        )

        return {
            "agent_name": AGENT_NAME,
            "success":    True,
            "confidence": pred["confidence"],
            "output": {
                "tumor_type":          pred["tumor_type"],
                "tumor_grade":         pred["tumor_grade"],
                "type_probabilities":  pred["type_probabilities"],
                "grade_probabilities": pred["grade_probabilities"],
                "clinical_urgency":    urgency,
                "tta_used":            pred["tta_used"],
            },
            "error": None,
        }

    # ── Error helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _fail(message: str) -> Dict[str, Any]:
        """Construct a standardised failure result."""
        print(f"[{AGENT_NAME}] FAILED: {message}")
        return {
            "agent_name": AGENT_NAME,
            "success":    False,
            "confidence": 0.0,
            "output": {
                "tumor_type":          None,
                "tumor_grade":         None,
                "type_probabilities":  None,
                "grade_probabilities": None,
                "clinical_urgency":    None,
                "tta_used":            False,
            },
            "error": message,
        }

    # ── Dunder ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"TumorClassificationAgent("
            f"device={self.device}, "
            f"predictor_ready={self._predictor is not None}, "
            f"tta={self.tta})"
        )
