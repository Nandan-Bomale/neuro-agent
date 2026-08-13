"""
agent.py
--------
TumorClassificationAgent — public interface for the Tumor Classification Agent.

This is the only file the Orchestrator imports.  All model loading, TTA,
and two-stage pipeline logic lives in inference.py.

Interface contract
------------------
Input  (state dict) :
    Required — one of:
        mri_slice_path   (str)         path to a 2-D MRI slice image
        mri_slice_array  (np.ndarray)  2-D / H×W×C numpy array

    Optional:
        vision_findings  (dict)        output from VisionAgent (for context logging)

Output (dict):
    {
        "agent_name":  "tumor_classification_agent",
        "success":     True,
        "confidence":  0.94,
        "output": {
            "tumor_type":          "glioma",
            "tumor_grade":         "grade_IV",     # None if not glioma
            "type_probabilities":  {"glioma": 0.94, "meningioma": 0.03, ...},
            "grade_probabilities": {"grade_II": 0.05, "grade_III": 0.18, "grade_IV": 0.77},
            "clinical_urgency":    "urgent",        # "urgent" | "routine" | "monitor"
            "tta_used":            True,
        },
        "error": None
    }

Clinical Urgency Mapping
------------------------
urgent  : glioma grade_IV  OR  glioma grade_III
monitor : glioma grade_II  OR  meningioma
routine : pituitary  OR  no_tumor

Usage
-----
    from agents.tumor_classification_agent.agent import TumorClassificationAgent

    agent = TumorClassificationAgent(
        type_ckpt_path  = "models/tumor_classifier/type_ensemble_best.pth",
        grade_ckpt_path = "models/tumor_classifier/grade_classifier_best.pth",
    )

    result = agent.run({
        "mri_slice_path": "data/sample_slice.jpg",
        "vision_findings": {...},   # optional VisionAgent output
    })

    print(result["output"]["tumor_type"])
    print(result["output"]["clinical_urgency"])
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

#: Default checkpoint paths — override in TumorClassificationAgent.__init__()
DEFAULT_TYPE_CKPT  = "models/tumor_classifier/type_ensemble_best.pth"
DEFAULT_GRADE_CKPT = "models/tumor_classifier/grade_classifier_best.pth"

AGENT_NAME = "tumor_classification_agent"


# ---------------------------------------------------------------------------
# Clinical urgency logic
# ---------------------------------------------------------------------------

_URGENCY_MAP: Dict[tuple, str] = {
    # (tumor_type, tumor_grade) -> urgency
    ("glioma",      "grade_IV"):  "urgent",
    ("glioma",      "grade_III"): "urgent",
    ("glioma",      "grade_II"):  "monitor",
    ("glioma",      None):        "monitor",   # grading unavailable
    ("meningioma",  None):        "monitor",
    ("pituitary",   None):        "routine",
    ("no_tumor",    None):        "routine",
}


def _clinical_urgency(tumor_type: str, tumor_grade: Optional[str]) -> str:
    """Derive clinical urgency from classification outputs.

    Args:
        tumor_type:  Stage-1 predicted class string.
        tumor_grade: Stage-2 predicted class string, or None.

    Returns:
        One of "urgent" | "monitor" | "routine".
    """
    # Normalise grade key (grade classifiers are only run for glioma)
    key = (tumor_type, tumor_grade)
    if key in _URGENCY_MAP:
        return _URGENCY_MAP[key]
    # Fallback for unknown types
    return "routine"


# ---------------------------------------------------------------------------
# TumorClassificationAgent
# ---------------------------------------------------------------------------

class TumorClassificationAgent:
    """Brain tumor type + glioma grade classification agent.

    Uses a two-stage pipeline:
        Stage 1 — EfficientNet-B4 + ResNet-50 + DenseNet-121 ensemble
                   → tumor type (4 classes)
        Stage 2 — EfficientNet-B4
                   → glioma grade (3 classes), only when Stage 1 = glioma

    Test-Time Augmentation (10 views) is applied at both stages by default.

    Args:
        type_ckpt_path:   Path to the Stage-1 ensemble checkpoint.
        grade_ckpt_path:  Path to the Stage-2 grade checkpoint.
        device:           Compute device.  Auto-detected if None.
        tta:              Enable Test-Time Augmentation (default True).
        tta_n:            Number of TTA views (default 10).

    Raises:
        FileNotFoundError: If a checkpoint cannot be found during ``run()``.
    """

    def __init__(
        self,
        type_ckpt_path:  str   = DEFAULT_TYPE_CKPT,
        grade_ckpt_path: str   = DEFAULT_GRADE_CKPT,
        device:          Optional[torch.device] = None,
        tta:             bool  = True,
        tta_n:           int   = 10,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device          = device
        self.type_ckpt_path  = type_ckpt_path
        self.grade_ckpt_path = grade_ckpt_path
        self.tta             = tta
        self.tta_n           = tta_n

        # Predictor is instantiated lazily inside run() on first call
        self._predictor: Optional[TumorPredictor] = None

        print(
            f"[TumorClassificationAgent] Initialised | "
            f"device={device} | TTA={tta} (n={tta_n})"
        )

    # ── Lazy initialisation ──────────────────────────────────────────────────

    def _ensure_predictor(self) -> None:
        """Instantiate TumorPredictor on first run() call (lazy init)."""
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
            state: Dictionary that MUST contain one of:
                   - ``"mri_slice_path"``  (str | Path) — path to a 2-D MRI slice
                   - ``"mri_slice_array"`` (np.ndarray) — H×W or H×W×C array

                   Optionally may contain:
                   - ``"vision_findings"`` (dict) — output from VisionAgent,
                     used for context logging only.

        Returns:
            Standard Orchestrator result dict:
            {
                "agent_name":  "tumor_classification_agent",
                "success":     True | False,
                "confidence":  float,
                "output": {
                    "tumor_type":          str,
                    "tumor_grade":         str | None,
                    "type_probabilities":  dict[str, float],
                    "grade_probabilities": dict[str, float] | None,
                    "clinical_urgency":    str,
                    "tta_used":            bool,
                },
                "error": None | str
            }
        """
        t_start = time.time()
        print(f"[{AGENT_NAME}] run() called.")

        # ── Validate input ────────────────────────────────────────────────────
        mri_source: Optional[Union[str, np.ndarray]] = None

        if "mri_slice_path" in state and state["mri_slice_path"] is not None:
            mri_source = str(state["mri_slice_path"])
            print(f"[{AGENT_NAME}] Input: path = {mri_source}")
        elif "mri_slice_array" in state and state["mri_slice_array"] is not None:
            mri_source = state["mri_slice_array"]
            print(
                f"[{AGENT_NAME}] Input: numpy array shape = {mri_source.shape}"
            )
        else:
            return self._error_result(
                "State must contain 'mri_slice_path' or 'mri_slice_array'."
            )

        # ── Log vision context ────────────────────────────────────────────────
        vision_findings = state.get("vision_findings")
        if vision_findings:
            print(
                f"[{AGENT_NAME}] Vision context: "
                f"tumour_detected={vision_findings.get('tumour_detected')}, "
                f"confidence={vision_findings.get('confidence_score')}"
            )

        # ── Run two-stage inference ───────────────────────────────────────────
        try:
            self._ensure_predictor()
            pred = self._predictor.predict(mri_source)
        except FileNotFoundError as exc:
            return self._error_result(str(exc))
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[{AGENT_NAME}] ERROR during inference:\n{tb}")
            return self._error_result(str(exc))

        # ── Derive clinical urgency ───────────────────────────────────────────
        urgency = _clinical_urgency(pred["tumor_type"], pred["tumor_grade"])

        elapsed = round(time.time() - t_start, 3)
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
    def _error_result(message: str) -> Dict[str, Any]:
        """Return a standardised failure result dict."""
        print(f"[{AGENT_NAME}] Error: {message}")
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

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        loaded = self._predictor is not None
        return (
            f"TumorClassificationAgent("
            f"device={self.device}, "
            f"type_ckpt='{Path(self.type_ckpt_path).name}', "
            f"grade_ckpt='{Path(self.grade_ckpt_path).name}', "
            f"predictor_loaded={loaded}, "
            f"tta={self.tta})"
        )
