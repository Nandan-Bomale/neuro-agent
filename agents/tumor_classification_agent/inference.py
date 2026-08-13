"""
inference.py
------------
Two-stage prediction pipeline for the Tumor Classification Agent.

Stage 1 — Predict tumor type from a 2-D MRI slice.
    Model : TumorTypeEnsemble (EfficientNet-B4 + ResNet-50 + DenseNet-121)
    Output: one of glioma / meningioma / notumor / pituitary

Stage 2 — Predict glioma grade (only when Stage 1 → glioma).
    Model : TumorGradeClassifier (EfficientNet-B4)
    Output: one of grade_II / grade_III / grade_IV

Test-Time Augmentation (TTA)
-----------------------------
When tta=True (default), 10 augmented views of the input are passed through
each model and the softmax outputs are averaged before argmax.
This consistently yields +1–2 % accuracy on unseen MRI slices.

Input formats accepted
-----------------------
• str / Path     — file path to any PIL-readable image (JPEG, PNG, TIFF…)
• np.ndarray     — H×W, H×W×1 (grayscale) or H×W×3 (RGB) array,
                   any dtype (auto-normalised to uint8)

Confidence
----------
Overall confidence = geometric mean of the type and grade max-probabilities
when both stages run, otherwise = type max-probability alone.

Usage
-----
    from agents.tumor_classification_agent.inference import TumorPredictor

    predictor = TumorPredictor(
        type_ckpt  = "models/tumor_classifier/type_ensemble_best.pth",
        grade_ckpt = "models/tumor_classifier/grade_classifier_best.pth",
    )

    result = predictor.predict("data/sample_slice.jpg")
    # result["tumor_type"]          -> "glioma"
    # result["tumor_grade"]         -> "grade_IV"
    # result["type_probabilities"]  -> {"glioma": 0.94, ...}
    # result["grade_probabilities"] -> {"grade_II": 0.05, ...}
    # result["confidence"]          -> 0.91
    # result["tta_used"]            -> True
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from agents.tumor_classification_agent.dataset import (
    _val_transforms,
    get_tta_transforms,
)
from agents.tumor_classification_agent.model import (
    TumorGradeClassifier,
    TumorTypeEnsemble,
    build_grade_classifier,
    build_type_ensemble,
)


# ---------------------------------------------------------------------------
# Default class lists  (fallback if no JSON sidecar exists)
# ---------------------------------------------------------------------------

# Must match the folder names in Training/ — confirmed: notumor (no underscore)
DEFAULT_TYPE_CLASSES  = ["glioma", "meningioma", "notumor", "pituitary"]
DEFAULT_GRADE_CLASSES = ["grade_II", "grade_III", "grade_IV"]

TTA_N = 10   # number of TTA augmented views


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_pil(source: Union[str, Path, np.ndarray]) -> Image.Image:
    """Load a 2-D MRI slice as a PIL Image (always returns RGB mode).

    Accepts:
        str / Path  — any PIL-readable file (JPEG, PNG, TIFF, BMP…)
        np.ndarray  — H×W, H×W×1, or H×W×3; any numeric dtype.
                      Normalised to [0, 255] uint8 when not already uint8.

    Returns:
        PIL Image in mode 'RGB'.

    Raises:
        TypeError  if source is not a recognised type.
        ValueError if array has an unsupported shape.
    """
    if isinstance(source, (str, Path)):
        img = Image.open(str(source))
        return img.convert("RGB")

    if isinstance(source, np.ndarray):
        arr = source.copy()
        # Normalise to [0, 255] uint8
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn) * 255.0
            arr = arr.astype(np.uint8)

        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[2] == 1:
            return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")

        raise ValueError(
            f"Unsupported array shape {arr.shape}. Expected H×W, H×W×1, or H×W×3."
        )

    raise TypeError(
        f"source must be a file path (str/Path) or numpy array, got {type(source).__name__}"
    )


# ---------------------------------------------------------------------------
# Class-name sidecar loader
# ---------------------------------------------------------------------------

def _load_class_names(checkpoint_path: str, default: List[str]) -> List[str]:
    """Try to load class names from a JSON sidecar next to the checkpoint.

    Convention:
        checkpoint: models/.../type_ensemble_best.pth
        sidecar:    models/.../type_ensemble_classes.json

    Strips trailing '_best', '_epNNN' or '_epochNNN' from the checkpoint stem
    to derive the sidecar stem.

    Returns:
        List of class name strings, or *default* if no sidecar is found.
    """
    ckpt   = Path(checkpoint_path)
    stem   = ckpt.stem
    # Strip common suffixes
    for suffix in ["_best"] + [f"_ep{i:03d}" for i in range(1, 200)] + [f"_epoch{i:03d}" for i in range(1, 200)]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    sidecar = ckpt.parent / f"{stem}_classes.json"
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text())
            return data.get("class_names", default)
        except Exception as exc:
            warnings.warn(f"Could not parse class sidecar {sidecar}: {exc}")
    return default


# ---------------------------------------------------------------------------
# Core TTA inference helper
# ---------------------------------------------------------------------------

def _tta_predict(
    model:       torch.nn.Module,
    img:         Image.Image,
    transforms:  List,
    device:      torch.device,
    is_ensemble: bool,
) -> torch.Tensor:
    """Run model on all TTA views and return averaged probability vector.

    Args:
        model:       Model in eval() mode.
        img:         PIL Image (RGB).
        transforms:  List of torchvision transform objects (one per TTA view).
        device:      Compute device.
        is_ensemble: True → use model.forward() which already returns probs.
                     False → apply softmax to raw logits.

    Returns:
        1-D probability tensor of shape (num_classes,), averaged over all views.
    """
    model.eval()
    accumulated: Optional[torch.Tensor] = None

    with torch.no_grad():
        for tfm in transforms:
            tensor = tfm(img).unsqueeze(0).to(device, non_blocking=True)
            if is_ensemble:
                probs = model(tensor).squeeze(0)           # forward() returns probs
            else:
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1).squeeze(0)

            accumulated = probs if accumulated is None else accumulated + probs

    return accumulated / len(transforms)   # type: ignore[operator]


# ---------------------------------------------------------------------------
# TumorPredictor
# ---------------------------------------------------------------------------

class TumorPredictor:
    """Two-stage tumor classification predictor.

    Models are loaded lazily on the first call to ``predict()``.

    Args:
        type_ckpt:   Path to Stage-1 ensemble checkpoint (.pth).
        grade_ckpt:  Path to Stage-2 grade classifier checkpoint (.pth).
        device:      Compute device.  Auto-detected if None.
        tta:         Enable Test-Time Augmentation (default True).
        tta_n:       Number of TTA views (default 10).
    """

    def __init__(
        self,
        type_ckpt:  str,
        grade_ckpt: str,
        device:     Optional[torch.device] = None,
        tta:        bool                   = True,
        tta_n:      int                    = TTA_N,
    ) -> None:
        self.type_ckpt   = type_ckpt
        self.grade_ckpt  = grade_ckpt
        self.tta         = tta
        self.tta_n       = tta_n

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Models loaded lazily
        self._type_model:  Optional[TumorTypeEnsemble]     = None
        self._grade_model: Optional[TumorGradeClassifier]  = None
        self._type_classes:  List[str] = []
        self._grade_classes: List[str] = []

        # Build transforms once at init
        self._tta_transforms   = get_tta_transforms(n_augmentations=tta_n)
        self._clean_transforms = [_val_transforms()]

        print(
            f"[TumorPredictor] Initialised | device={device} | "
            f"TTA={tta} (n={tta_n})"
        )

    # ── Lazy model loaders ───────────────────────────────────────────────────

    def _load_type_model(self) -> None:
        if self._type_model is not None:
            return
        ckpt = self.type_ckpt
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"[TumorPredictor] Type checkpoint not found: {ckpt}\n"
                "Train Stage 1 first:\n"
                "  python -m agents.tumor_classification_agent.train "
                "--stage type --data-path data/tumor_classification/type/ "
                "--save-dir models/tumor_classifier/"
            )
        self._type_classes = _load_class_names(ckpt, DEFAULT_TYPE_CLASSES)
        self._type_model   = build_type_ensemble(
            num_classes=len(self._type_classes),
            checkpoint_path=ckpt,
            device=self.device,
        )
        self._type_model.eval()
        print(f"[TumorPredictor] Type model ready | classes={self._type_classes}")

    def _load_grade_model(self) -> None:
        if self._grade_model is not None:
            return
        ckpt = self.grade_ckpt
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"[TumorPredictor] Grade checkpoint not found: {ckpt}\n"
                "Train Stage 2 first:\n"
                "  python -m agents.tumor_classification_agent.train "
                "--stage grade --data-path data/tumor_classification/grade/ "
                "--brats-path data/raw/BraTS2020_TrainingData/ "
                "--save-dir models/tumor_classifier/"
            )
        self._grade_classes = _load_class_names(ckpt, DEFAULT_GRADE_CLASSES)
        self._grade_model   = build_grade_classifier(
            num_classes=len(self._grade_classes),
            checkpoint_path=ckpt,
            device=self.device,
        )
        self._grade_model.eval()
        print(f"[TumorPredictor] Grade model ready | classes={self._grade_classes}")

    # ── Public predict ───────────────────────────────────────────────────────

    def predict(
        self,
        source:    Union[str, Path, np.ndarray],
        run_grade: Optional[bool] = None,
    ) -> Dict:
        """Run the full two-stage prediction on one MRI slice.

        Args:
            source:    File path or 2-D numpy array representing one MRI slice.
            run_grade: Override Stage-2 execution.  If None (default), Stage 2
                       runs automatically when Stage 1 predicts "glioma".

        Returns:
            dict with keys:
                tumor_type          (str)
                tumor_grade         (str | None)
                type_probabilities  (dict[str, float])
                grade_probabilities (dict[str, float] | None)
                confidence          (float)
                tta_used            (bool)
        """
        self._load_type_model()
        img      = _load_pil(source)
        tfm_list = self._tta_transforms if self.tta else self._clean_transforms

        # ── Stage 1: Tumor type ───────────────────────────────────────────────
        type_probs = _tta_predict(
            model=self._type_model,
            img=img,
            transforms=tfm_list,
            device=self.device,
            is_ensemble=True,           # forward() already returns averaged softmax
        )
        type_probs_np   = type_probs.cpu().numpy()
        type_idx        = int(np.argmax(type_probs_np))
        tumor_type      = self._type_classes[type_idx]
        type_confidence = float(type_probs_np[type_idx])

        type_probabilities = {
            cls: round(float(p), 4)
            for cls, p in zip(self._type_classes, type_probs_np)
        }

        # ── Stage 2: Glioma grade ─────────────────────────────────────────────
        should_grade = (
            run_grade if run_grade is not None else (tumor_type == "glioma")
        )
        tumor_grade         = None
        grade_probabilities = None
        confidence          = round(type_confidence, 4)

        if should_grade:
            self._load_grade_model()
            grade_probs = _tta_predict(
                model=self._grade_model,
                img=img,
                transforms=tfm_list,
                device=self.device,
                is_ensemble=False,
            )
            grade_probs_np  = grade_probs.cpu().numpy()
            grade_idx       = int(np.argmax(grade_probs_np))
            tumor_grade     = self._grade_classes[grade_idx]
            grade_confidence = float(grade_probs_np[grade_idx])

            grade_probabilities = {
                cls: round(float(p), 4)
                for cls, p in zip(self._grade_classes, grade_probs_np)
            }

            # Geometric mean of both stages' max-probabilities
            confidence = round(float(np.sqrt(type_confidence * grade_confidence)), 4)

        return {
            "tumor_type":          tumor_type,
            "tumor_grade":         tumor_grade,
            "type_probabilities":  type_probabilities,
            "grade_probabilities": grade_probabilities,
            "confidence":          confidence,
            "tta_used":            self.tta,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    @property
    def type_classes(self) -> List[str]:
        """Type class names (populated after first predict call)."""
        return self._type_classes or DEFAULT_TYPE_CLASSES

    @property
    def grade_classes(self) -> List[str]:
        """Grade class names (populated after first grade predict call)."""
        return self._grade_classes or DEFAULT_GRADE_CLASSES

    def __repr__(self) -> str:
        return (
            f"TumorPredictor("
            f"device={self.device}, "
            f"type_loaded={self._type_model is not None}, "
            f"grade_loaded={self._grade_model is not None}, "
            f"tta={self.tta}, tta_n={self.tta_n})"
        )
