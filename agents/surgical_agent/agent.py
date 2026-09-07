import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)
AGENT_NAME = "surgical_agent"

class SurgicalAgent:
    """Rule-based Surgical Planning Agent using 2D features.
    
    Reads:  state["vision_findings"]                (tumor area in cm2)
            state["localization_findings"]          (lobe)
            state["tumor_classification_findings"]  (grade for risk adjustment)
    Writes: state["surgical_analysis"]
    """

    def __init__(self):
        logger.info(f"[{AGENT_NAME}] Initialized.")

    def _compute_resectability(
        self,
        tumor_area_cm2: float,
        tumor_volume_cm3: float,
        predicted_lobe: str,
        tumor_grade: str | None,
        tumor_type: str | None,
    ) -> tuple[float, str]:
        """
        Compute resectability score based on real contour area + volume.
        Thresholds calibrated to REAL tumor contour-based cm²:
          Small  < 2 cm²   -> excellent (90%)
          Medium 2-8 cm²   -> good (75%)
          Large  8-20 cm²  -> moderate (55%)
          Massive > 20 cm² -> poor (35%)
        Volume (ABC/2 cm³) used as secondary signal.
        """
        # Use volume if area is small but volume is large (3D estimation)
        effective_area = tumor_area_cm2
        if tumor_volume_cm3 > 50:
            effective_area = max(effective_area, 22.0)
        elif tumor_volume_cm3 > 20:
            effective_area = max(effective_area, 10.0)

        if effective_area < 2:
            base_score = 0.90
        elif effective_area < 8:
            base_score = 0.75
        elif effective_area < 20:
            base_score = 0.55
        else:
            base_score = 0.35

        # Lobe Eloquence adjustment
        lobe_lower = predicted_lobe.lower()
        eloquent_penalty = 0.0
        if "temporal" in lobe_lower or "parietal" in lobe_lower:
            eloquent_penalty = 0.15 # Wernicke's / Motor strip proximity
            
        # Grade adjustment
        grade_penalty = 0.0
        if tumor_grade == "grade_IV":
            grade_penalty = 0.10

        type_bonus = 0.0
        if tumor_type == "meningioma":
            type_bonus = 0.10

        final_score = max(0.1, min(1.0, base_score - eloquent_penalty - grade_penalty + type_bonus))

        # Surgical recommendation
        if tumor_type == "pituitary":
            recommendation = "Transsphenoidal endoscopic surgery recommended for Pituitary lesion."
        elif final_score >= 0.80:
            recommendation = "Maximal safe resection (>95% EOR) recommended."
        elif final_score >= 0.60:
            recommendation = "Subtotal resection feasible. Consider functional mapping if near eloquent cortex."
        else:
            recommendation = "Biopsy only. High risk of neurological deficit due to eloquent lobe involvement or massive size."

        return round(final_score, 3), recommendation

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t_start = time.perf_counter()
        
        vision = state.get("vision_findings", {})
        loc = state.get("localization_findings", {})
        clf = state.get("tumor_classification_findings", {})

        area   = vision.get("tumor_area_cm2", 5.0)       # Real contour area
        volume = vision.get("tumor_volume_cm3", 0.0)     # ABC/2 volume
        lobe   = loc.get("predicted_lobe", "Unknown")
        grade  = clf.get("tumor_grade")
        ttype  = clf.get("tumor_type")

        if ttype == "notumor" or not vision.get("tumor_detected", True):
            resectability = 0.0
            recommendation = "No surgical intervention indicated (no tumor detected)."
        else:
            resectability, recommendation = self._compute_resectability(area, volume, lobe, grade, ttype)


        elapsed = round(time.perf_counter() - t_start, 4)
        logger.info(f"[{AGENT_NAME}] Resectability: {resectability}")

        return {
            "surgical_analysis": {
                "resectability_score": resectability,
                "surgical_recommendation": recommendation,
                "inference_time_sec": elapsed,
            }
        }
