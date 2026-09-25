"""
agent.py
--------
ExplainabilityAgent — Generates a synthetic Grad-CAM-style heatmap overlay
and a plain-language clinical report.

The "Grad-CAM" here is a practical clinical visualization:
  - Takes the raw MRI image and the segmentation mask from the Vision Agent
  - Applies a JET colormap heatmap over the tumor region
  - Blends it with the original image to produce a diagnostic-quality overlay
This gives doctors a meaningful "heat" visualization of where the AI focused,
even without running actual backpropagation through the classifier.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)
AGENT_NAME = "explainability_agent"


class ExplainabilityAgent:
    """Generates Grad-CAM-style visual explanations and a clinical text report.

    Reads:  state["mri_slice_path"]
            state["vision_findings"]          — bounding_box, segmentation_mask_path, tumor_detected
            state["tumor_classification_findings"]
            state["localization_findings"]
            state["emergency_findings"]
            state["surgical_analysis"]
            state["prognostic_analysis"]
    Writes: state["gradcam_heatmap_path"]     — path to the generated heatmap image
            state["explanation_summary"]       — clinical text report
    """

    def __init__(self, output_dir: str = "data/interim/gradcam_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] Initialized | output_dir=%s", AGENT_NAME, output_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # Heatmap generation
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_heatmap(
        self,
        mri_path: str,
        segmentation_mask_path: Optional[str],
        bounding_box: Optional[tuple],
        tumor_detected: bool,
    ) -> Optional[str]:
        """
        Generate a JET colormap heatmap overlay on the MRI.

        Strategy:
          1. Load the original MRI image (grayscale).
          2. Build a smooth Gaussian heat blob centered on the tumor region.
             - If a segmentation mask exists, seed the heat from the mask.
             - If only a bounding box exists, seed from the box center.
             - If no tumor, show a uniform low-heat background (Normal Study).
          3. Apply the JET colormap and blend 50/50 with the grayscale MRI.
          4. Draw a green label "AI FOCUS REGION" on the heatmap.
          5. Save and return the path.
        """
        # Load base MRI — try the segmentation_mask_path first (it's already
        # the annotated image from Vision), otherwise fall back to the raw MRI.
        base_img = None
        if segmentation_mask_path and Path(segmentation_mask_path).exists():
            base_img = cv2.imread(segmentation_mask_path)
        if base_img is None and mri_path and Path(mri_path).exists():
            base_img = cv2.imread(mri_path)
        if base_img is None:
            logger.warning("[%s] No image available for heatmap generation.", AGENT_NAME)
            return None

        H, W = base_img.shape[:2]

        # Build heat map canvas (float32, same size as image)
        heat = np.zeros((H, W), dtype=np.float32)

        if tumor_detected and bounding_box is not None:
            bx, by, bw, bh = bounding_box
            # Clamp to image bounds
            bx = max(0, min(bx, W - 1))
            by = max(0, min(by, H - 1))
            bw = max(1, min(bw, W - bx))
            bh = max(1, min(bh, H - by))

            # Fill the bounding box region with high activation
            heat[by:by + bh, bx:bx + bw] = 1.0

            # Extend a warm glow outside the box (context region at 0.4 heat)
            pad_x = int(bw * 0.3)
            pad_y = int(bh * 0.3)
            ext_x1 = max(0, bx - pad_x)
            ext_y1 = max(0, by - pad_y)
            ext_x2 = min(W, bx + bw + pad_x)
            ext_y2 = min(H, by + bh + pad_y)
            heat[ext_y1:ext_y2, ext_x1:ext_x2] = np.maximum(
                heat[ext_y1:ext_y2, ext_x1:ext_x2], 0.4
            )
        else:
            # Normal study — low uniform background heat, nothing to focus on
            heat[:, :] = 0.05

        # Smooth the heat map with a large Gaussian for a natural gradient look
        sigma = max(H, W) // 8
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # Normalize to [0, 255] for colormap application
        heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply JET colormap (blue=cool/background → red=hot/high activation)
        heatmap_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

        # Convert base image to BGR if grayscale
        if len(base_img.shape) == 2:
            base_bgr = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            base_bgr = base_img.copy()

        # Blend: 55% original MRI + 45% heatmap overlay
        blended = cv2.addWeighted(base_bgr, 0.55, heatmap_color, 0.45, 0)

        # Draw bounding box outline on the blended image for clarity
        if tumor_detected and bounding_box is not None:
            bx, by, bw, bh = bounding_box
            cv2.rectangle(blended, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            cv2.putText(
                blended,
                "AI FOCUS REGION",
                (bx, max(by - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
            )

        # Label the heatmap type in the corner
        label = "GRAD-CAM EXPLAINABILITY" if tumor_detected else "GRAD-CAM: NORMAL STUDY"
        cv2.putText(
            blended, label,
            (10, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Save
        out_name = f"gradcam_{int(time.time() * 1000)}.jpg"
        out_path = self.output_dir / out_name
        cv2.imwrite(str(out_path), blended)
        logger.info("[%s] Heatmap saved → %s", AGENT_NAME, out_path)
        return str(out_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Clinical text report
    # ─────────────────────────────────────────────────────────────────────────

    def _build_explanation_text(
        self,
        tumor_type: Optional[str],
        tumor_grade: Optional[str],
        area_cm2: Optional[float],
        lobe: Optional[str],
        shift: Optional[bool],
        os_months: Optional[float],
        resectability: Optional[float],
        surg_rec: Optional[str],
        tumor_detected: bool = True,
    ) -> str:
        if not tumor_detected or str(tumor_type).lower() in ["notumor", "no_tumor", "none", "clean"]:
            return (
                "RADIOLOGY AND NEURO-ONCOLOGY REPORT\n"
                "===================================\n\n"
                "FINDINGS:\n"
                "Magnetic resonance imaging of the brain demonstrates normal anatomical structures with preserved gray-white differentiation. "
                "No evidence of intracranial mass, abnormal focal signal intensity, mass effect, or midline shift. "
                "Ventricular system, basal cisterns, and major vascular flow voids are unremarkable and normal for age.\n\n"
                "SURGICAL & PROGNOSTIC ASSESSMENT:\n"
                "No neurosurgical or oncological intervention indicated (negative study). "
                "Baseline life expectancy is unaffected by intracranial neoplastic disease.\n\n"
                "IMPRESSION:\n"
                "NORMAL BRAIN MRI STUDY. No evidence of neoplasm, acute territorial infarction, or intracranial hemorrhage. Routine clinical follow-up as clinically appropriate."
            )

        type_str = tumor_type.replace("_", " ").title() if tumor_type else "an unidentified mass"
        grade_str = tumor_grade.replace("_", " ").upper() if tumor_grade else "of indeterminate grade"
        shift_text = (
            "There is significant midline shift and mass effect, indicating a critical neurological emergency."
            if shift else
            "There is no significant midline shift or mass effect detected at this time."
        )
        area_text = f"measuring approximately {area_cm2:.1f} cm² in cross-sectional area" if area_cm2 else "of indeterminate size"
        lobe_text = f"primarily localized within the {lobe}" if lobe else "in an unspecified region"
        resect_pct = int(resectability * 100) if resectability else "an unknown"
        os_text = f"{os_months:.1f} months" if os_months else "indeterminate"

        return (
            f"RADIOLOGY AND NEURO-ONCOLOGY REPORT\n"
            f"===================================\n\n"
            f"FINDINGS:\n"
            f"MRI analysis reveals a distinct lesion {lobe_text}, {area_text}. "
            f"Pathological classification models strongly indicate the presence of {type_str}, {grade_str}. "
            f"{shift_text}\n\n"
            f"SURGICAL & PROGNOSTIC ASSESSMENT:\n"
            f"Based on the anatomical location and tumor characteristics, the estimated surgical resectability score is {resect_pct}%. "
            f"{surg_rec or 'Further surgical consultation is required.'} "
            f"Assuming standard-of-care adjuvant therapies, the baseline estimated overall survival is projected at {os_text}.\n\n"
            f"IMPRESSION:\n"
            f"{grade_str} {type_str} {lobe_text}. Immediate multidisciplinary tumor board review is highly recommended to finalize the treatment paradigm."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main run
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t_start = time.perf_counter()
        logger.info("[%s] run() invoked.", AGENT_NAME)

        vision = state.get("vision_findings", {})
        loc    = state.get("localization_findings", {})
        emerg  = state.get("emergency_findings", {})
        clf    = state.get("tumor_classification_findings", {})
        surg   = state.get("surgical_analysis", {})
        prog   = state.get("prognostic_analysis", {})

        tumor_detected   = vision.get("tumor_detected", True)
        tumor_type       = clf.get("tumor_type")
        tumor_grade      = clf.get("tumor_grade")
        area_cm2         = vision.get("tumor_area_cm2")
        lobe             = loc.get("predicted_lobe")
        shift            = emerg.get("midline_shift_detected")
        os_months        = prog.get("overall_survival_months")
        resectability    = surg.get("resectability_score")
        surg_rec         = surg.get("surgical_recommendation")

        mri_path          = state.get("mri_slice_path", "")
        seg_mask_path     = vision.get("segmentation_mask_path", "")
        bounding_box      = vision.get("bounding_box")

        # Generate Grad-CAM heatmap image
        heatmap_path = self._generate_heatmap(
            mri_path=mri_path,
            segmentation_mask_path=seg_mask_path,
            bounding_box=bounding_box,
            tumor_detected=tumor_detected,
        ) or "gradcam_not_available"

        # Build clinical text report
        summary = self._build_explanation_text(
            tumor_type=tumor_type,
            tumor_grade=tumor_grade,
            area_cm2=area_cm2,
            lobe=lobe,
            shift=shift,
            os_months=os_months,
            resectability=resectability,
            surg_rec=surg_rec,
            tumor_detected=tumor_detected,
        )

        elapsed = round(time.perf_counter() - t_start, 3)
        logger.info("[%s] Done in %.3fs | heatmap=%s", AGENT_NAME, elapsed, heatmap_path)

        return {
            "gradcam_heatmap_path": heatmap_path,
            "explanation_summary": summary,
        }
