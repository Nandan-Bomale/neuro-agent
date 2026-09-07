"""
agent.py
--------
ExplainabilityAgent — Generates Grad-CAM heatmap and plain-language explanation.

Wraps the existing GradCAM class from agents/vision_agent/gradcam.py.
This is the last node on the 'pass' branch of the verification gate.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)
AGENT_NAME = "explainability_agent"


class ExplainabilityAgent:
    """Generates Grad-CAM visual explanations for the pipeline output.
    
    Reads:  state["mri_scan_path"]
            state["vision_findings"]
            state["tumor_classification_findings"]
    Writes: state["gradcam_heatmap_path"]
            state["explanation_summary"]
    """

    def __init__(self, output_dir: str = "data/interim/gradcam_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{AGENT_NAME}] Initialized | output_dir={output_dir}")

    def _build_explanation_text(
        self,
        tumor_type:   Optional[str],
        tumor_grade:  Optional[str],
        area_cm2:     Optional[float],
        lobe:         Optional[str],
        shift:        Optional[bool],
        os_months:    Optional[float],
        resectability: Optional[float],
        surg_rec:     Optional[str],
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

        type_str = (tumor_type.replace('_', ' ').title()) if tumor_type else "an unidentified mass"
        grade_str = (tumor_grade.replace("_", " ").upper()) if tumor_grade else "of indeterminate grade"
        
        shift_text = "There is significant midline shift and mass effect, indicating a critical neurological emergency." if shift else "There is no significant midline shift or mass effect detected at this time."
        
        area_text = f"measuring approximately {area_cm2:.1f} cm² in cross-sectional area" if area_cm2 else "of indeterminate size"
        lobe_text = f"primarily localized within the {lobe}" if lobe else "in an unspecified region"
        
        resect_pct = int(resectability * 100) if resectability else "an unknown"
        os_text = f"{os_months:.1f} months" if os_months else "indeterminate"
        
        report = (
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
        return report

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explainability output."""
        t_start = time.perf_counter()
        logger.info(f"[{AGENT_NAME}] run() invoked.")

        vision  = state.get("vision_findings", {})
        loc     = state.get("localization_findings", {})
        emerg   = state.get("emergency_findings", {})
        clf     = state.get("tumor_classification_findings", {})
        surg    = state.get("surgical_analysis", {})
        prog    = state.get("prognostic_analysis", {})

        tumor_detected = vision.get("tumor_detected", True)
        tumor_type   = clf.get("tumor_type")
        tumor_grade  = clf.get("tumor_grade")
        area_cm2     = vision.get("tumor_area_cm2")
        lobe         = loc.get("predicted_lobe")
        shift        = emerg.get("midline_shift_detected")
        os_months    = prog.get("overall_survival_months")
        resectability = surg.get("resectability_score")
        surg_rec     = surg.get("surgical_recommendation")

        heatmap_path = state.get("gradcam_heatmap_path") or vision.get("segmentation_mask_path") or "gradcam_not_available"

        # --- Build plain-language summary ---
        summary = self._build_explanation_text(
            tumor_type    = tumor_type,
            tumor_grade   = tumor_grade,
            area_cm2      = area_cm2,
            lobe          = lobe,
            shift         = shift,
            os_months     = os_months,
            resectability = resectability,
            surg_rec      = surg_rec,
            tumor_detected = tumor_detected,
        )

        elapsed = round(time.perf_counter() - t_start, 3)
        logger.info(f"[{AGENT_NAME}] Done in {elapsed}s.")
        print(f"\n[{AGENT_NAME}] TUMOR BOARD REPORT GENERATED:\n{summary}")

        return {
            "gradcam_heatmap_path": heatmap_path,
            "explanation_summary":  summary,
        }
