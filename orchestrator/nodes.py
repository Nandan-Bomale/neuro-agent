"""
NeuroAgent — LangGraph Node Functions
=======================================
Thin wrapper functions that plug each agent class into the LangGraph graph.

Rules:
  • Each function signature: (state: NeuroAgentState) -> dict[str, Any]
  • Return ONLY the fields this node writes — LangGraph merges them into state.
  • NO ML logic here. Every node just calls agent.run(state) and returns the result.
  • Agent imports are lazy (inside the function) so missing agents surface as
    RuntimeError at call time, not at import time — keeps the orchestrator
    importable even while individual agents are still being built.

Integration checklist (Week 10–11):
  [x] VisionAgent          — agents/vision_agent/agent.py
  [x] TumorClassAgent      — agents/tumor_classification_agent/agent.py  ✅ DONE
  [ ] RadiogenomicsAgent   — agents/radiogenomics_agent/agent.py
  [ ] SurgicalAgent        — agents/surgical_agent/agent.py
  [ ] PrognosticAgent      — agents/prognostic_agent/agent.py
  [ ] ClinicalTrialAgent   — agents/clinical_trial_agent/agent.py
  [ ] NeuroOncologistAgent — agents/neuro_oncologist_agent/agent.py
  [x] VerificationAgent    — agents/verification_agent/agent.py  ✅ DONE
"""

from __future__ import annotations

import logging
from typing import Any

from orchestrator.state import NeuroAgentState

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _not_yet_implemented(agent_name: str, module_path: str) -> RuntimeError:
    """Return a clear RuntimeError for agents not yet built."""
    return RuntimeError(
        f"{agent_name} is not yet implemented.\n"
        f"  Expected: {module_path}\n"
        f"  Action:   Create a class '{agent_name}' with a .run(state: dict) -> dict method."
    )


# ── Agent nodes ────────────────────────────────────────────────────────────────


def vision_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: Vision Agent
    Runs tumour detection + segmentation on the MRI scan.

    Reads:  state["mri_scan_path"]
    Writes: state["vision_findings"]  (dict with confidence, mask, regions, label)
    """
    logger.info("[vision_node] Starting VisionAgent | scan=%s", state.get("mri_scan_path"))

    try:
        from agents.vision_agent.agent import VisionAgent  # noqa: PLC0415
    except (ImportError, AttributeError) as exc:
        raise _not_yet_implemented(
            "VisionAgent", "agents/vision_agent/agent.py"
        ) from exc

    agent = VisionAgent()
    vision_result = agent.run(
        mri_scan_path=state["mri_scan_path"]
    )

    logger.info(
        "[vision_node] Done | confidence=%.4f | label=%s",
        vision_result.confidence_score,
        "tumour_detected" if vision_result.tumour_detected else "no_tumour",
    )
    return {
        "vision_findings": vision_result.to_dict(),
        "mri_slice_path": vision_result.mri_slice_path,
    }


def tumor_classification_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: Tumor Classification Agent
    Identifies tumor TYPE (glioma/meningioma/pituitary/no_tumor) using
    EfficientNet-B4 + ResNet50 + DenseNet121 ensemble, then classifies
    glioma GRADE (II/III/IV) with a second EfficientNet-B4.

    Reads:  state["vision_findings"]  (for context)
            state["mri_slice_path"]   (2D slice extracted by Vision Agent)
    Writes: state["tumor_classification_findings"]
    """
    logger.info("[tumor_classification_node] Starting TumorClassificationAgent")

    try:
        from agents.tumor_classification_agent.agent import TumorClassificationAgent  # noqa: PLC0415
    except (ImportError, AttributeError) as exc:
        raise _not_yet_implemented(
            "TumorClassificationAgent",
            "agents/tumor_classification_agent/agent.py",
        ) from exc

    agent = TumorClassificationAgent()
    result = agent.run(dict(state))

    # The agent returns {"agent_name":..., "success":..., "output": {...}, "confidence":...}
    # We need to map its "output" sub-dict to state["tumor_classification_findings"]
    agent_output = result.get("output", {})
    confidence   = result.get("confidence", 0.0)

    tumor_classification_findings = {
        "tumor_type":          agent_output.get("tumor_type"),
        "tumor_grade":         agent_output.get("tumor_grade"),
        "type_probabilities":  agent_output.get("type_probabilities"),
        "grade_probabilities": agent_output.get("grade_probabilities"),
        "clinical_urgency":    agent_output.get("clinical_urgency"),
        "confidence":          confidence,
        "tta_used":            agent_output.get("tta_used", False),
    }

    logger.info(
        "[tumor_classification_node] Done | type=%s | grade=%s | confidence=%.4f",
        tumor_classification_findings.get("tumor_type", "?"),
        tumor_classification_findings.get("tumor_grade", "N/A"),
        tumor_classification_findings.get("confidence", -1),
    )
    return {"tumor_classification_findings": tumor_classification_findings}


def radiogenomics_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: Radiogenomics Agent
    Predicts IDH mutation and MGMT methylation status directly from the MRI.

    Reads:  state["mri_scan_path"], state["tumor_classification_findings"]
    Writes: state["radiogenomics_findings"]
    """
    logger.info("[radiogenomics_node] Starting RadiogenomicsAgent")

    try:
        from agents.radiogenomics_agent.agent import RadiogenomicsAgent  # noqa: PLC0415
    except (ImportError, AttributeError) as exc:
        raise _not_yet_implemented(
            "RadiogenomicsAgent", "agents/radiogenomics_agent/agent.py"
        ) from exc

    agent = RadiogenomicsAgent()
    result = agent.run(dict(state))
    return result


def surgical_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: Surgical Planning Agent
    Assesses resectability and eloquent area proximity.

    Reads:  state["mri_scan_path"], state["vision_findings"]
    Writes: state["surgical_analysis"]
    """
    logger.info("[surgical_node] Starting SurgicalAgent")

    try:
        from agents.surgical_agent.agent import SurgicalAgent  # noqa: PLC0415
    except (ImportError, AttributeError) as exc:
        raise _not_yet_implemented(
            "SurgicalAgent", "agents/surgical_agent/agent.py"
        ) from exc

    agent = SurgicalAgent()
    result = agent.run(dict(state))
    return result


def prognostic_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: Prognostic Agent (DeepSurv)
    Calculates overall survival and progression-free survival.

    Reads:  state["tumor_classification_findings"], state["radiogenomics_findings"], state["patient_data"]
    Writes: state["prognostic_analysis"]
    """
    logger.info("[prognostic_node] Starting PrognosticAgent")

    try:
        from agents.prognostic_agent.agent import PrognosticAgent  # noqa: PLC0415
    except (ImportError, AttributeError) as exc:
        raise _not_yet_implemented(
            "PrognosticAgent", "agents/prognostic_agent/agent.py"
        ) from exc

    agent = PrognosticAgent()
    result = agent.run(dict(state))
    return result


def clinical_trial_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: Clinical Trial Agent
    Scrapes clinicaltrials.gov for matched trials.

    Reads:  state["tumor_classification_findings"], state["radiogenomics_findings"]
    Writes: state["clinical_trials"]
    """
    logger.info("[clinical_trial_node] Starting ClinicalTrialAgent")

    try:
        from agents.clinical_trial_agent.agent import ClinicalTrialAgent  # noqa: PLC0415
    except (ImportError, AttributeError) as exc:
        raise _not_yet_implemented(
            "ClinicalTrialAgent", "agents/clinical_trial_agent/agent.py"
        ) from exc

    agent = ClinicalTrialAgent()
    result = agent.run(dict(state))
    return result


def neuro_oncologist_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: Neuro-Oncologist Agent (LoRA LLM)
    Generates personalized treatment plan based on NCCN guidelines.

    Reads:  state["tumor_classification_findings"], state["radiogenomics_findings"], state["prognostic_analysis"], state["surgical_analysis"]
    Writes: state["neuro_oncologist_plan"], state["overall_confidence"]
    """
    logger.info("[neuro_oncologist_node] Starting NeuroOncologistAgent")

    try:
        from agents.neuro_oncologist_agent.agent import NeuroOncologistAgent  # noqa: PLC0415
    except (ImportError, AttributeError) as exc:
        raise _not_yet_implemented(
            "NeuroOncologistAgent", "agents/neuro_oncologist_agent/agent.py"
        ) from exc

    agent = NeuroOncologistAgent()
    result = agent.run(dict(state))

    logger.info(
        "[neuro_oncologist_node] Done | overall_confidence=%.4f",
        result.get("overall_confidence", -1),
    )
    return result


def verification_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: Verification Agent
    Checks overall_confidence against CONFIDENCE_THRESHOLD.
    Routes to explainability (pass) or human review (fail).

    Reads:  state["overall_confidence"] | state["vision_findings"]["confidence"]
    Writes: state["verification_status"], state["requires_human_review"],
            state["verification_notes"], state["confidence_threshold"]
    """
    logger.info("[verification_node] Starting VerificationAgent")

    # VerificationAgent is already implemented — direct import (no try/except needed)
    from agents.verification_agent.agent import VerificationAgent  # noqa: PLC0415

    agent = VerificationAgent()
    result = agent.run(dict(state))

    logger.info(
        "[verification_node] Done | status=%s | confidence_threshold=%.2f",
        result.get("verification_status"),
        result.get("confidence_threshold", -1),
    )
    return result


def explainability_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: Explainability Agent
    Produces a Grad-CAM heatmap and plain-language explanation.
    Only reached on the 'pass' branch after verification.

    Reads:  state["mri_scan_path"], state["vision_findings"]
    Writes: state["gradcam_heatmap_path"], state["explanation_summary"],
            state["pipeline_status"] = "complete"
    """
    logger.info("[explainability_node] Starting ExplainabilityAgent")

    try:
        from agents.explainability_agent.agent import ExplainabilityAgent  # noqa: PLC0415
    except (ImportError, AttributeError) as exc:
        raise _not_yet_implemented(
            "ExplainabilityAgent", "agents/explainability_agent/agent.py"
        ) from exc

    agent = ExplainabilityAgent()
    result = agent.run(dict(state))

    logger.info(
        "[explainability_node] Done | heatmap=%s",
        result.get("gradcam_heatmap_path", "?"),
    )

    # Explainability is the last node on the pass branch — mark pipeline complete
    return {**result, "pipeline_status": "complete"}


def human_review_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: Human Review Flag  (no agent — pure orchestration logic)
    Reached when verification_status == 'human_review'.
    Sets the pipeline status and surfaces the reason for review.
    The result is surfaced to the backend / frontend for escalation.

    Reads:  state["verification_notes"]
    Writes: state["human_review_reason"], state["pipeline_status"]
    """
    reason = state.get("verification_notes", "Confidence below threshold.")
    logger.warning("[human_review_node] Case flagged for HUMAN REVIEW | reason=%s", reason)

    return {
        "pipeline_status": "human_review_required",
        "human_review_reason": reason,
    }
