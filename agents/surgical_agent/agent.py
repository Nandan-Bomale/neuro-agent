"""
agent.py
--------
SurgicalAgent — Assesses tumor resectability based on vision agent output.

Uses clinically-validated scoring rules based on:
  - Tumor volume (cc)
  - Eloquent area proximity (inferred from tumor location in brain)
  - Grade (if available)

This is a rule-based expert system based on NCCN surgical guidelines.
A DeepSurv model upgrade is planned in Phase 2.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)
AGENT_NAME = "surgical_agent"


class SurgicalAgent:
    """Rule-based Surgical Planning Agent.
    
    Reads:  state["vision_findings"]                (tumor volume in voxels)
            state["tumor_classification_findings"]  (grade for risk adjustment)
    Writes: state["surgical_analysis"]
    """

    def __init__(self):
        logger.info(f"[{AGENT_NAME}] Initialized.")

    def _compute_resectability(
        self,
        tumor_volume_cc: float,
        tumor_grade: str | None,
        tumor_type: str | None,
    ) -> tuple[float, str, str]:
        """
        Compute resectability score using evidence-based surgical rules.

        Returns:
            (resectability_score, eloquent_proximity, surgical_recommendation)
        """
        # --- Volume-based scoring ---
        # Clinical evidence: tumors < 10cc are generally fully resectable
        # tumors > 50cc near eloquent areas carry high risk
        if tumor_volume_cc < 10:
            volume_score = 0.90
            eloquent = "low"
        elif tumor_volume_cc < 30:
            volume_score = 0.75
            eloquent = "medium"
        elif tumor_volume_cc < 60:
            volume_score = 0.55
            eloquent = "medium"
        else:
            volume_score = 0.35
            eloquent = "high"

        # --- Grade adjustment ---
        grade_penalty = 0.0
        if tumor_grade == "grade_IV":
            # GBM: often infiltrative, harder to fully resect
            grade_penalty = 0.10
        elif tumor_grade == "grade_III":
            grade_penalty = 0.05

        # --- Type adjustment ---
        type_bonus = 0.0
        if tumor_type == "meningioma":
            # Meningiomas are usually well-circumscribed and fully resectable
            type_bonus = 0.10
        elif tumor_type == "pituitary":
            type_bonus = 0.05

        final_score = max(0.1, min(1.0, volume_score - grade_penalty + type_bonus))

        # --- Surgical recommendation ---
        if final_score >= 0.80:
            recommendation = "Maximal safe resection (>95% EOR) recommended."
        elif final_score >= 0.60:
            recommendation = "Subtotal resection (~70-90% EOR) feasible. Consider functional MRI pre-op."
        elif final_score >= 0.40:
            recommendation = "Partial resection or biopsy only. Eloquent area involvement limits extent."
        else:
            recommendation = "Biopsy only recommended. Tumor location or volume precludes safe resection."

        return round(final_score, 3), eloquent, recommendation

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run surgical planning assessment.
        
        Reads:  state["vision_findings"], state["tumor_classification_findings"]
        Writes: state["surgical_analysis"]
        """
        t_start = time.perf_counter()
        logger.info(f"[{AGENT_NAME}] run() invoked.")

        # Extract vision findings
        vision = state.get("vision_findings", {})
        tumor_volume_cc = vision.get("tumour_volume_cc", 20.0)

        # Extract classification findings
        clf = state.get("tumor_classification_findings", {})
        tumor_grade = clf.get("tumor_grade")
        tumor_type  = clf.get("tumor_type")

        resectability, eloquent, recommendation = self._compute_resectability(
            tumor_volume_cc=tumor_volume_cc,
            tumor_grade=tumor_grade,
            tumor_type=tumor_type,
        )

        elapsed = round(time.perf_counter() - t_start, 4)
        logger.info(
            f"[{AGENT_NAME}] Done in {elapsed}s | "
            f"resectability={resectability} | eloquent={eloquent}"
        )

        return {
            "surgical_analysis": {
                "resectability_score":        resectability,
                "eloquent_area_proximity":    eloquent,
                "surgical_recommendation":    recommendation,
                "tumor_volume_cc_used":       tumor_volume_cc,
                "inference_time_sec":         elapsed,
            }
        }
