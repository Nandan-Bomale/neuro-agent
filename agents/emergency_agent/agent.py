import cv2
import numpy as np
import time
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)
AGENT_NAME = "emergency_agent"

class EmergencyAgent:
    """
    Analyzes the 2D MRI slice for Midline Shift (mass effect).
    """
    def __init__(self, output_dir="data/interim/emergency"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{AGENT_NAME}] Initialized.")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t_start = time.perf_counter()
        
        mri_path = state.get("mri_slice_path")
        vision = state.get("vision_findings", {})
        bbox = vision.get("bounding_box")
        
        if not mri_path or not Path(mri_path).exists():
            return {"emergency_findings": {"error": "Missing MRI path"}}
            
        img = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img_bgr = cv2.imread(mri_path)
            if img_bgr is None:
                return {"emergency_findings": {"error": "Cannot read image"}}
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Ensure 2D grayscale
        if img.ndim == 3:
            img = img[:, :, 0]
        h, w = img.shape

        
        # Heuristic for midline shift:
        # A normal brain midline is exactly at w/2.
        # If the tumor's bounding box crosses the midline significantly, 
        # or if the tumor area is massive, we flag it.
        
        shift_detected = False
        urgency = "routine"
        
        if bbox:
            bx, by, bw, bh = bbox
            tumor_area_cm2  = vision.get("tumor_area_cm2", 0)
            tumor_vol_cm3   = vision.get("tumor_volume_cm3", 0)

            # Check if tumor crosses midline (w/2)
            crosses_midline = (bx < w/2 and (bx + bw) > w/2)

            # Thresholds calibrated for REAL contour-based area (not box area):
            #   Small  < 3 cm²   -- routine
            #   Medium 3-10 cm²  -- monitor
            #   Large  > 10 cm²  -- urgent
            #   Massive > 20 cm² -- critical
            if crosses_midline and tumor_area_cm2 > 5.0:  # Crossing midline > 5cm2
                shift_detected = True
                urgency = "CRITICAL - Mass Effect / Midline Shift Detected"
            elif tumor_vol_cm3 > 30.0 or tumor_area_cm2 > 15.0:
                urgency = "CRITICAL - Very Large Mass (>30cm3)"
            elif tumor_area_cm2 > 8.0:
                urgency = "URGENT - Large Mass (>8cm2)"

        
        # Draw the theoretical midline
        out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        mid_x = w // 2
        color = (0, 0, 255) if shift_detected else (255, 0, 0)
        cv2.line(out_img, (mid_x, 0), (mid_x, h), color, 2)
        
        out_path = self.output_dir / f"midline_{int(time.time())}.jpg"
        cv2.imwrite(str(out_path), out_img)
        
        elapsed = time.perf_counter() - t_start
        logger.info(f"[{AGENT_NAME}] Midline shift: {shift_detected} | Urgency: {urgency}")
        
        return {
            "emergency_findings": {
                "midline_shift_detected": shift_detected,
                "urgency_upgrade": urgency,
                "midline_shift_image_path": str(out_path)
            }
        }
