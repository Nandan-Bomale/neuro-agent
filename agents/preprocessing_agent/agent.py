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
        
        # Reject non-brain (e.g. empty or non-brain scan)
        if brain_coverage_ratio < 0.06:
            logger.warning("[%s] Brain coverage %.1f%% < 6%% -- not a brain MRI", AGENT_NAME, brain_coverage_ratio * 100)
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
        
        logger.info("[%s] Done %.3fs | seq=%s | brain_cov=%.1f%% | skull=%s",
                    AGENT_NAME, elapsed, mri_sequence, brain_coverage_ratio * 100, skull_stripped)
                    
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