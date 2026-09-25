import cv2
import time
import logging
import numpy as np
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)
AGENT_NAME = "preprocessing_agent"

def _classify_sequence(brain_pixels: np.ndarray) -> str:
    if len(brain_pixels) < 100:
        return 'T1'
    bp = brain_pixels.astype(float)
    brain_mean = float(bp.mean())
    brain_std = float(max(bp.std(), 1.0))
    bright_ratio = float(np.mean(bp > brain_mean + 2.0 * brain_std))
    skewness = float(np.mean(((bp - brain_mean) / brain_std) ** 3))
    
    if bright_ratio > 0.08 and brain_mean > 80:
        return 'T1CE'
    elif brain_std > 45 and skewness > 0.8:
        return 'FLAIR'
    elif brain_mean < 70 and brain_std > 35:
        return 'T2'
    else:
        return 'T1'

def _strip_borders(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    out = img.copy()
    for rows in [slice(0, max(1, int(H * 0.04))), slice(min(H - 1, int(H * 0.98)), H)]:
        region = out[rows, :]
        mean_val = float(region.mean()) if region.size > 0 else 0
        if mean_val > 180 or (0 < mean_val < 5):
            out[rows, :] = 0
    return out

class PreprocessingAgent:
    def __init__(self, output_dir="data/interim/preprocessing"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] Initialized.", AGENT_NAME)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t_start = time.perf_counter()
        mri_path = state.get("mri_slice_path")
        if not mri_path or not Path(mri_path).exists():
            return {"preprocessing_findings": {"error": "Missing or invalid mri_slice_path"}}

        img = cv2.imread(mri_path)
        if img is None:
            return {"preprocessing_findings": {"error": "Cannot read image"}}

        original_path = mri_path
        H_full, W_full = img.shape[:2]
        total_pixels = H_full * W_full

        # ── GATE 1: Color saturation check ─────────────────────────────────────
        # Real MRI scans come from a magnetic scanner — they are always pure
        # grayscale. Even when saved as JPEG color (3 channels), all three
        # channels are identical copies of the grayscale. Natural photos,
        # even when they look faded, still contain residual color tint.
        # Threshold: MRIs < 10, natural photos typically 15–80+.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_saturation = float(hsv[:, :, 1].mean())
        if mean_saturation > 15.0:
            logger.warning(
                "[%s] REJECTED: mean_saturation=%.1f > 15 — color photo, not a MRI.",
                AGENT_NAME, mean_saturation
            )
            return {
                "mri_slice_path": original_path,
                "preprocessing_findings": {
                    "error": (
                        "Invalid Input: This appears to be a natural photograph, not a brain MRI scan. "
                        "Please upload a valid grayscale brain MRI image."
                    ),
                    "is_brain": False,
                    "brain_coverage_ratio": 0.0,
                    "mri_sequence": "unknown",
                    "skull_stripped": False,
                    "inference_time_sec": round(time.perf_counter() - t_start, 4)
                }
            }

        # ── GATE 2: Hue variety check ──────────────────────────────────────────
        # A true grayscale MRI has virtually zero hue variation — all pixels
        # cluster around hue=0 because they're gray. A natural photo (even a
        # desaturated one) has sky, vegetation, concrete, skin — producing many
        # different hue values. Count unique hue values in the image.
        # MRIs: typically < 5 unique hues. Natural photos: > 20.
        hue_channel = hsv[:, :, 0]
        unique_hues = len(np.unique(hue_channel[hsv[:, :, 1] > 10]))  # only count pixels that have some color
        if unique_hues > 20:
            logger.warning(
                "[%s] REJECTED: unique_hues=%d > 20 — color-varied image, not a MRI.",
                AGENT_NAME, unique_hues
            )
            return {
                "mri_slice_path": original_path,
                "preprocessing_findings": {
                    "error": (
                        "Invalid Input: This image contains too many distinct colors to be a grayscale medical scan. "
                        "Please upload a valid brain MRI scan."
                    ),
                    "is_brain": False,
                    "brain_coverage_ratio": 0.0,
                    "mri_sequence": "unknown",
                    "skull_stripped": False,
                    "inference_time_sec": round(time.perf_counter() - t_start, 4)
                }
            }

        # ── GATE 3: Edge density check ──────────────────────────────────────────
        # Brain MRIs have smooth tissue gradients with very few strong edges.
        # Natural photographs are full of high-frequency structural edges.
        # Canny edge coverage > 8% is a strong signal this is NOT a medical scan.
        gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_raw, threshold1=40, threshold2=120)
        edge_density = float(np.count_nonzero(edges)) / total_pixels
        if edge_density > 0.08:
            logger.warning(
                "[%s] REJECTED: edge_density=%.3f > 0.08 — complex natural image, not a MRI.",
                AGENT_NAME, edge_density
            )
            return {
                "mri_slice_path": original_path,
                "preprocessing_findings": {
                    "error": (
                        "Invalid Input: This image has too many sharp edges to be a medical scan. "
                        "Please upload a valid brain MRI scan."
                    ),
                    "is_brain": False,
                    "brain_coverage_ratio": 0.0,
                    "mri_sequence": "unknown",
                    "skull_stripped": False,
                    "inference_time_sec": round(time.perf_counter() - t_start, 4)
                }
            }

        # 1. Clean borders/watermarks
        img = _strip_borders(img)

        # 2. CLAHE contrast enhancement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        # 3. Brain contour finding
        _, thresh = cv2.threshold(gray_clahe, 15, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return {
                "mri_slice_path": original_path,
                "preprocessing_findings": {
                    "error": "No brain contour found",
                    "is_brain": False,
                    "mri_sequence": "unknown",
                    "skull_stripped": False,
                    "inference_time_sec": time.perf_counter() - t_start
                }
            }

        largest_cnt = max(cnts, key=cv2.contourArea)
        brain_contour_area = cv2.contourArea(largest_cnt)
        brain_coverage_ratio = float(brain_contour_area / max(total_pixels, 1))

        # ── GATE 3: Brain coverage ratio ────────────────────────────────────────
        if brain_coverage_ratio < 0.06:
            logger.warning(
                "[%s] REJECTED: brain_coverage=%.1f%% < 6%% — not a brain MRI.",
                AGENT_NAME, brain_coverage_ratio * 100
            )
            return {
                "mri_slice_path": original_path,
                "preprocessing_findings": {
                    "error": "Not a brain MRI - image rejected",
                    "is_brain": False,
                    "brain_coverage_ratio": round(brain_coverage_ratio, 4),
                    "mri_sequence": "unknown",
                    "skull_stripped": False,
                    "inference_time_sec": time.perf_counter() - t_start
                }
            }

        brain_mask_raw = np.zeros_like(gray_clahe)
        cv2.drawContours(brain_mask_raw, [largest_cnt], -1, 255, -1)
        brain_pixels = gray_clahe[brain_mask_raw > 0]
        mri_sequence = _classify_sequence(brain_pixels)

        # 4. Skull stripping
        stripped_img = img.copy()
        skull_stripped = False
        kernel = np.ones((12, 12), np.uint8)
        eroded = cv2.erode(brain_mask_raw, kernel, iterations=1)
        if cv2.countNonZero(eroded) > total_pixels * 0.05:
            brain_mask = cv2.dilate(eroded, np.ones((8, 8), np.uint8), iterations=1)
            stripped_img = cv2.bitwise_and(img, img, mask=brain_mask)
            skull_stripped = True
        else:
            brain_mask = brain_mask_raw

        out_name = f"preprocessed_{int(time.time())}.jpg"
        out_path = self.output_dir / out_name
        cv2.imwrite(str(out_path), stripped_img)
        elapsed = time.perf_counter() - t_start

        logger.info(
            "[%s] Done %.3fs | seq=%s | brain_cov=%.1f%% | skull=%s",
            AGENT_NAME, elapsed, mri_sequence, brain_coverage_ratio * 100, skull_stripped
        )

        return {
            "mri_slice_path": str(out_path),
            "preprocessing_findings": {
                "preprocessed_mri_path": str(out_path),
                "skull_stripped": skull_stripped,
                "is_brain": True,
                "mri_sequence": mri_sequence,
                "brain_coverage_ratio": round(brain_coverage_ratio, 4),
                "inference_time_sec": elapsed
            }
        }