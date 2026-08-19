"""
agent.py
--------
PrognosticAgent — Evidence-based survival estimation for brain tumor patients.

Uses a clinical scoring model based on validated prognostic factors from
the TCGA and CGGA (Chinese Glioma Genome Atlas) survival studies:
  - Tumor grade (strongest predictor)
  - IDH mutation status (second strongest predictor)
  - MGMT methylation (predicts chemotherapy response)
  - Patient age
  - Tumor volume
  - Surgical resectability

Phase 2 upgrade: Replace with a trained DeepSurv neural network.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
AGENT_NAME = "prognostic_agent"


# ── Survival lookup tables from published clinical studies ─────────────────
# Source: TCGA GBM/LGG cohort, Stupp et al., CGGA studies
# Values are median OS/PFS in months

GRADE_SURVIVAL = {
    "grade_II":  {"os": 96.0,  "pfs": 60.0},   # Low-grade glioma: 8 years OS
    "grade_III": {"os": 24.0,  "pfs": 14.0},   # Anaplastic glioma: 2 years OS
    "grade_IV":  {"os": 15.0,  "pfs": 7.5},    # GBM: 14-15 months OS (Stupp 2005)
}

TYPE_SURVIVAL = {
    "meningioma": {"os": 180.0, "pfs": 120.0},  # Mostly benign, >15yr survival
    "pituitary":  {"os": 240.0, "pfs": 180.0},  # Excellent prognosis
    "glioma":     None,                          # Uses GRADE_SURVIVAL
    "notumor":    {"os": 360.0, "pfs": 360.0},  # No tumor found
}

# Multipliers applied to base survival based on molecular markers
IDH_MUTANT_MULTIPLIER  = 2.5    # IDH mutant = much better prognosis
IDH_WILDTYPE_MULTIPLIER = 1.0

MGMT_METHYLATED_OS_BONUS   = 6.0   # Months added: responds to Temozolomide
MGMT_UNMETHYLATED_OS_BONUS = 0.0


class PrognosticAgent:
    """Evidence-based Prognostic Agent for brain tumor survival estimation.
    
    Reads:  state["tumor_classification_findings"]
            state["radiogenomics_findings"]
            state["patient_data"]
            state["surgical_analysis"]
    Writes: state["prognostic_analysis"]
    """

    def __init__(self):
        logger.info(f"[{AGENT_NAME}] Initialized.")

    def _estimate_survival(
        self,
        tumor_type:  str,
        tumor_grade: Optional[str],
        idh_status:  str,
        mgmt_status: str,
        age:         float,
        resectability: float,
    ) -> tuple[float, float, str]:
        """Compute OS and PFS using validated clinical scoring rules.
        
        Returns:
            (overall_survival_months, progression_free_survival_months, risk_category)
        """
        # --- Get base survival from tumor type/grade ---
        type_entry = TYPE_SURVIVAL.get(tumor_type)
        if type_entry is not None:
            base_os  = type_entry["os"]
            base_pfs = type_entry["pfs"]
        else:
            # Default to glioma grade-based survival
            grade_entry = GRADE_SURVIVAL.get(tumor_grade, GRADE_SURVIVAL["grade_IV"])
            base_os  = grade_entry["os"]
            base_pfs = grade_entry["pfs"]

        # --- IDH adjustment ---
        idh_mult = IDH_MUTANT_MULTIPLIER if idh_status == "mutant" else IDH_WILDTYPE_MULTIPLIER
        base_os  *= idh_mult
        base_pfs *= idh_mult

        # --- MGMT adjustment (adds months due to chemo response) ---
        base_os  += MGMT_METHYLATED_OS_BONUS if mgmt_status == "methylated" else MGMT_UNMETHYLATED_OS_BONUS

        # --- Age adjustment (>65 carries ~20% worse prognosis) ---
        if age and age > 65:
            base_os  *= 0.80
            base_pfs *= 0.80
        elif age and age < 40:
            base_os  *= 1.15
            base_pfs *= 1.10

        # --- Surgical adjustment (full resection = +20% survival) ---
        if resectability >= 0.80:
            base_os  *= 1.20
            base_pfs *= 1.15
        elif resectability < 0.40:
            base_os  *= 0.85
            base_pfs *= 0.85

        # --- Risk category ---
        if base_os >= 60:
            risk = "low"
        elif base_os >= 18:
            risk = "moderate"
        else:
            risk = "high"

        return round(base_os, 1), round(base_pfs, 1), risk

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run prognostic survival estimation.
        
        Reads:  tumor_classification_findings, radiogenomics_findings,
                patient_data, surgical_analysis
        Writes: prognostic_analysis
        """
        t_start = time.perf_counter()
        logger.info(f"[{AGENT_NAME}] run() invoked.")

        # Extract all inputs
        clf     = state.get("tumor_classification_findings", {})
        radio   = state.get("radiogenomics_findings", {})
        patient = state.get("patient_data", {})
        surg    = state.get("surgical_analysis", {})

        tumor_type   = clf.get("tumor_type",  "glioma")
        tumor_grade  = clf.get("tumor_grade", "grade_IV")
        idh_status   = radio.get("idh_mutation_status",    "wildtype")
        mgmt_status  = radio.get("mgmt_methylation_status","unmethylated")
        age          = float(patient.get("age", 55))
        resectability = float(surg.get("resectability_score", 0.75))

        os_months, pfs_months, risk = self._estimate_survival(
            tumor_type    = tumor_type,
            tumor_grade   = tumor_grade,
            idh_status    = idh_status,
            mgmt_status   = mgmt_status,
            age           = age,
            resectability = resectability,
        )

        elapsed = round(time.perf_counter() - t_start, 4)
        logger.info(
            f"[{AGENT_NAME}] Done in {elapsed}s | "
            f"OS={os_months}mo | PFS={pfs_months}mo | risk={risk}"
        )

        return {
            "prognostic_analysis": {
                "overall_survival_months":           os_months,
                "progression_free_survival_months":  pfs_months,
                "risk_category":                     risk,
                "key_factors": {
                    "grade":            tumor_grade,
                    "idh_status":       idh_status,
                    "mgmt_status":      mgmt_status,
                    "age":              age,
                    "resectability":    resectability,
                },
                "inference_time_sec": elapsed,
                "source": "evidence_based_clinical_scoring",
            }
        }
