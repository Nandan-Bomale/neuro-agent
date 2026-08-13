"""
inference.py
------------
Load trained weights and run a two-stage prediction pipeline:

  Stage 1 — Predict tumor type (glioma / meningioma / pituitary / no_tumor)
  Stage 2 — Predict glioma grade (grade_II / grade_III / grade_IV)
             [only executed when Stage 1 = glioma]

Accepts either a file-path (str/Path) or a pre-loaded numpy array as input.

Test-Time Augmentation (TTA)
----------------------------
When tta=True (the default), inference runs 10 augmented views of the input
image and averages the softmax outputs before argmax.  This consistently
improves accuracy on unseen MRI slices by ~1–2 %.

Usage
-----
    from agents.tumor_classification_agent.inference import TumorPredictor

    predictor = TumorPredictor(
        type_ckpt  = "models/tumor_classifier/type_ensemble_best.pth",
        grade_ckpt = "models/tumor_classifier/grade_classifier_best.pth",
    )

    result = predictor.predict("path/to/mri_slice.jpg")
    # result["tumor_type"]          -> "glioma"
    # result["tumor_grade"]         -> "grade_IV"   (or None)
    # result["type_probabilities"]  -> {"glioma": 0.94, ...}
    # result["grade_probabilities"] -> {"grade_II": 0.05, ...}
    # result["confidence"]          -> 0.94
    # result["tta_used"]            -> True
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from agents.tumor_classification_agent.dataset import get_tta_transforms, _val_transforms
from agents.tumor_classification_agent.model import (
    build_grade_classifier,
    build_type_ensemble,
)


# ---------------------------------------------------------------------------
# Default class names (fallback if no metadata JSON is found)
# ---------------------------------------------------------------------------

DEFAULT_TYPE_CLASSES  = ["glioma", "meningioma", "notumor", "pituitary"]
DEFAULT_GRADE_CLASSES = ["grade_II", "grade_III", "grade_IV"]

TTA_N = 10  # number of TTA augmented views


# ---------------------------------------------------------------------------
# Image loading utilities
# ---------------------------------------------------------------------------

def _load_image(source: Union[str, Path, np.ndarray]) -> Image.Image:
    """Load an MRI slice from a file path or numpy array.

    Grayscale arrays / single-channel images are accepted; RGB conversion is
    handled downstream by GrayscaleToRGB in the dataset transforms.

    Args:
        source: File path (str / Path) or H×W or H×W×C numpy array.

    Returns:
        PIL Image in mode 'L' (grayscale) or 'RGB'.
    """
    if isinstance(source, (str, Path)):
        img = Image.open(str(source))
    elif isinstance(source, np.ndarray):
        arr = source
        if arr.dtype != np.uint8:
            # Normalise to [0, 255]
            arr = arr - arr.min()
            if arr.max() > 0:
                arr = (arr / arr.max() * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            img = Image.fromarray(arr, mode="L")
        elif arr.ndim == 3 and arr.shape[2] == 1:
            img = Image.fromarray(arr[:, :, 0], mode="L")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray(arr, mode="RGB")
        else:
            raise ValueError(
                f"Unsupported array shape for MRI slice: {arr.shape}. "
                "Expected H×W, H×W×1, or H×W×3."
            )
    else:
        raise TypeError(
            f"source must be a file path or numpy array, got {type(source)}"
        )
    return img


def _load_class_names(
    checkpoint_path: str,
    default: List[str],
) -> List[str]:
    """Try to load class names from a JSON sidecar file next to the checkpoint.

    Falls back to *default* if no sidecar is found.

    Convention: checkpoint 'foo_best.pth' → sidecar 'foo_classes.json'.
    """
    ckpt_path   = Path(checkpoint_path)
    # Strip _best / _epochXXX suffix
    stem = ckpt_path.stem
    for suffix in ("_best", *[f"_epoch{i:03d}" for i in range(1, 200)]):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    sidecar = ckpt_path.parent / f"{stem}_classes.json"
    if sidecar.exists():
        data = json.loads(sidecar.read_text())
        return data.get("class_names", default)
    return default


# ---------------------------------------------------------------------------
# TTA helper
# ---------------------------------------------------------------------------

def _run_tta(
    model:      torch.nn.Module,
    img:        Image.Image,
    transforms: list,
    device:     torch.device,
    is_ensemble: bool,
) -> torch.Tensor:
    """Run model on *n* augmented views of *img* and return averaged probabilities.

    Args:
        model:       Model in eval mode.
        img:         PIL Image.
        transforms:  List of transform objects (one per TTA view).
        device:      Compute device.
        is_ensemble: If True, use ``model.predict_proba``; else softmax logits.

    Returns:
        Averaged probability tensor of shape ``(num_classes,)``.
    """
    accumulated = None

    for tfm in transforms:
        tensor = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            if is_ensemble:
                probs = model.predict_proba(tensor).squeeze(0)
            else:
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1).squeeze(0)

        accumulated = probs if accumulated is None else accumulated + probs

    return accumulated / len(transforms)


# ---------------------------------------------------------------------------
# TumorPredictor
# ---------------------------------------------------------------------------

class TumorPredictor:
    """Two-stage tumor classification predictor.

    Stage 1 — Ensemble predicts tumor type (4 classes).
    Stage 2 — EfficientNet-B4 predicts glioma grade (3 classes), called only
               when Stage 1 output is "glioma".

    Both models are loaded lazily on the first call to ``predict``.

    Args:
        type_ckpt:   Path to the Stage-1 ensemble checkpoint (.pth).
        grade_ckpt:  Path to the Stage-2 grade classifier checkpoint (.pth).
        device:      Compute device.  Auto-detected if None.
        tta:         Whether to use Test-Time Augmentation (default True).
        tta_n:       Number of TTA augmented views (default 10).
    """

    def __init__(
        self,
        type_ckpt:  str,
        grade_ckpt: str,
        device:     Optional[torch.device] = None,
        tta:        bool = True,
        tta_n:      int  = TTA_N,
    ) -> None:
        self.type_ckpt   = type_ckpt
        self.grade_ckpt  = grade_ckpt
        self.tta         = tta
        self.tta_n       = tta_n

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self._type_model:  Optional[torch.nn.Module] = None
        self._grade_model: Optional[torch.nn.Module] = None
        self._type_classes:  List[str] = []
        self._grade_classes: List[str] = []

        # Pre-compute TTA transforms once
        self._tta_transforms  = get_tta_transforms(n_augmentations=tta_n)
        self._clean_transform = _val_transforms()

        print(
            f"[TumorPredictor] device={device} | TTA={tta} (n={tta_n})"
        )

    # ── Lazy loaders ─────────────────────────────────────────────────────────

    def _ensure_type_model(self) -> None:
        if self._type_model is not None:
            return
        if not Path(self.type_ckpt).exists():
            raise FileNotFoundError(
                f"[TumorPredictor] Type checkpoint not found: {self.type_ckpt}\n"
                "Train Stage 1 first:  python -m agents.tumor_classification_agent.train "
                "--stage type --data-path <path> --save-dir models/tumor_classifier/"
            )
        self._type_classes = _load_class_names(self.type_ckpt, DEFAULT_TYPE_CLASSES)
        self._type_model   = build_type_ensemble(
            num_classes=len(self._type_classes),
            checkpoint_path=self.type_ckpt,
            device=self.device,
        )
        self._type_model.eval()
        print(
            f"[TumorPredictor] Type model loaded | "
            f"classes={self._type_classes}"
        )

    def _ensure_grade_model(self) -> None:
        if self._grade_model is not None:
            return
        if not Path(self.grade_ckpt).exists():
            raise FileNotFoundError(
                f"[TumorPredictor] Grade checkpoint not found: {self.grade_ckpt}\n"
                "Train Stage 2 first:  python -m agents.tumor_classification_agent.train "
                "--stage grade --data-path <path> --save-dir models/tumor_classifier/"
            )
        self._grade_classes = _load_class_names(self.grade_ckpt, DEFAULT_GRADE_CLASSES)
        self._grade_model   = build_grade_classifier(
            num_classes=len(self._grade_classes),
            checkpoint_path=self.grade_ckpt,
            device=self.device,
        )
        self._grade_model.eval()
        print(
            f"[TumorPredictor] Grade model loaded | "
            f"classes={self._grade_classes}"
        )

    # ── Core predict ─────────────────────────────────────────────────────────

    def predict(
        self,
        source:      Union[str, Path, np.ndarray],
        run_grade:   Optional[bool] = None,
    ) -> Dict:
        """Run the two-stage prediction pipeline on one MRI slice.

        Args:
            source:    File path (str / Path) or 2-D numpy array representing
                       a single MRI slice.
            run_grade: Override whether to run Stage 2.  If None (default),
                       Stage 2 runs automatically when Stage 1 predicts "glioma".

        Returns:
            dict with keys:
                tumor_type         (str)
                tumor_grade        (str | None)
                type_probabilities (dict[str, float])
                grade_probabilities (dict[str, float] | None)
                confidence         (float)  — max probability across classes
                tta_used           (bool)
        """
        self._ensure_type_model()

        img      = _load_image(source)
        tfm_list = self._tta_transforms if self.tta else [self._clean_transform]

        # ── Stage 1: Tumor Type ───────────────────────────────────────────────
        type_probs_tensor = _run_tta(
            model=self._type_model,
            img=img,
            transforms=tfm_list,
            device=self.device,
            is_ensemble=True,
        )
        type_probs_np  = type_probs_tensor.cpu().numpy()
        type_idx       = int(np.argmax(type_probs_np))
        tumor_type     = self._type_classes[type_idx]
        type_confidence = float(type_probs_np[type_idx])

        type_probabilities = {
            cls: round(float(p), 4)
            for cls, p in zip(self._type_classes, type_probs_np)
        }

        # ── Stage 2: Glioma Grade ─────────────────────────────────────────────
        should_grade = (
            run_grade if run_grade is not None else (tumor_type == "glioma")
        )
        tumor_grade         = None
        grade_probabilities = None
        confidence          = type_confidence

        if should_grade:
            self._ensure_grade_model()

            grade_probs_tensor = _run_tta(
                model=self._grade_model,
                img=img,
                transforms=tfm_list,
                device=self.device,
                is_ensemble=False,
            )
            grade_probs_np = grade_probs_tensor.cpu().numpy()
            grade_idx      = int(np.argmax(grade_probs_np))
            tumor_grade    = self._grade_classes[grade_idx]
            grade_conf     = float(grade_probs_np[grade_idx])

            grade_probabilities = {
                cls: round(float(p), 4)
                for cls, p in zip(self._grade_classes, grade_probs_np)
            }

            # Overall confidence = geometric mean of type and grade confidence
            confidence = round(float(np.sqrt(type_confidence * grade_conf)), 4)

        return {
            "tumor_type":          tumor_type,
            "tumor_grade":         tumor_grade,
            "type_probabilities":  type_probabilities,
            "grade_probabilities": grade_probabilities,
            "confidence":          round(confidence, 4),
            "tta_used":            self.tta,
        }

    def __repr__(self) -> str:
        return (
            f"TumorPredictor("
            f"device={self.device}, "
            f"type_ckpt='{Path(self.type_ckpt).name}', "
            f"grade_ckpt='{Path(self.grade_ckpt).name}', "
            f"tta={self.tta})"
        )
