import time
import logging
from typing import Dict, Any, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)
AGENT_NAME = "localization_agent"

class LocalizationAgent:
    """
    Predicts the anatomical lobe / location of the tumor using simulated 
    atlas-based registration (normalized brain-centric coordinates).
    """
    def __init__(self):
        logger.info(f"[{AGENT_NAME}] Initialized Atlas-Based Localization.")

    def _get_brain_bbox(self, img_gray: np.ndarray) -> Tuple[int, int, int, int]:
        # Morphological operations to find the actual brain boundary
        _, thresh = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            return cv2.boundingRect(c)
        return 0, 0, img_gray.shape[1], img_gray.shape[0]

    def _map_to_atlas(self, nx: float, ny: float, nw: float, nh: float, is_pituitary: bool) -> str:
        if is_pituitary:
            return "Sella Turcica (Pituitary Fossa)"
            
        regions = []
        center_x = nx + nw / 2
        center_y = ny + nh / 2
        
        # Radiological convention: Left side of image is Right Hemisphere
        hemisphere = "Right" if center_x < 0.5 else "Left"
        
        # Check Midline crossing
        if nx < 0.45 and (nx + nw) > 0.55:
            regions.append("Crosses Midline (Corpus Callosum invasion risk)")
            hemisphere = "Bilateral"
            
        # Check Deep Brain
        if 0.35 < center_x < 0.65 and 0.4 < center_y < 0.6:
            regions.append("Basal Ganglia / Thalamus (Deep Brain structure)")
            
        # Y-axis mapping (Anterior to Posterior)
        if center_y < 0.2:
            regions.append("Prefrontal Cortex")
        elif 0.2 <= center_y < 0.45:
            regions.append("Frontal Lobe (near Motor Cortex)")
        elif 0.45 <= center_y < 0.7:
            if center_x < 0.25 or center_x > 0.75:
                regions.append("Temporal Lobe")
            else:
                regions.append("Parietal Lobe")
        elif 0.7 <= center_y < 0.85:
            regions.append("Occipital Lobe")
        else:
            regions.append("Cerebellum / Brainstem region")
            
        primary = regions[-1]
        other_notes = ", ".join(regions[:-1])
        
        result = f"{hemisphere} {primary}"
        if other_notes:
            result += f" [{other_notes}]"
            
        return result

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t_start = time.perf_counter()
        
        vision = state.get("vision_findings", {})
        bbox = vision.get("bounding_box")
        mri_path = state.get("mri_slice_path")
        tumor_type = state.get("tumor_classification_findings", {}).get("tumor_type", "")
        
        if not vision.get("tumor_detected", True) or not bbox:
            logger.info(f"[{AGENT_NAME}] No tumor detected -- normal brain parenchyma.")
            return {
                "localization_findings": {
                    "predicted_lobe": "No lesion detected (normal brain parenchyma)",
                    "bounding_box": None,
                    "normalized_coords": None,
                    "brain_bbox": None,
                    "image_dims": None
                }
            }
            
        if not mri_path:
            return {"localization_findings": {"error": "Missing image path"}}
            
        img = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"localization_findings": {"error": "Cannot read image"}}
            
        bx, by, bw, bh = bbox
        
        # 1. Atlas Registration (Normalize against Brain Bounding Box)
        br_x, br_y, br_w, br_h = self._get_brain_bbox(img)
        
        # Normalize tumor bbox relative to brain
        nx = max(0.0, (bx - br_x) / br_w)
        ny = max(0.0, (by - br_y) / br_h)
        nw = min(1.0 - nx, bw / br_w)
        nh = min(1.0 - ny, bh / br_h)
        
        # 2. Atlas Lookup
        is_pituitary = (tumor_type == "pituitary")
        predicted_lobe = self._map_to_atlas(nx, ny, nw, nh, is_pituitary)
        
        elapsed = time.perf_counter() - t_start
        logger.info(f"[{AGENT_NAME}] Atlas mapping complete: {predicted_lobe}")
        
        return {
            "localization_findings": {
                "predicted_lobe": predicted_lobe,
                "bounding_box": bbox,
                "normalized_coords": (nx, ny, nw, nh),
                "brain_bbox": (br_x, br_y, br_w, br_h),
                "image_dims": (img.shape[1], img.shape[0])
            }
        }
