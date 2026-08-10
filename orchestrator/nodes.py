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
  [ ] VisionAgent          — agents/vision_agent/agent.py
  [ ] ClinicalHistoryAgent — agents/clinical_history_agent/agent.py
  [ ] RAGLiteratureAgent   — agents/rag_literature_agent/agent.py
  [ ] ReportAgent          — agents/report_generation_agent/agent.py
  [ ] ExplainabilityAgent  — agents/explainability_agent/agent.py
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
    result = agent.run(dict(state))

    logger.info(
        "[vision_node] Done | confidence=%.4f | label=%s",
        result.get("vision_findings", {}).get("confidence", -1),
        result.get("vision_findings", {}).get("prediction_label", "?"),
    )
    return result


def clinical_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: Clinical History Agent
    Reasons over patient metadata and how it fits the vision findings.

    Reads:  state["patient_data"], state["vision_findings"]
    Writes: state["clinical_analysis"]
    """
    logger.info("[clinical_node] Starting ClinicalHistoryAgent")

    try:
        from agents.clinical_history_agent.agent import ClinicalHistoryAgent  # noqa: PLC0415
    except (ImportError, AttributeError) as exc:
        raise _not_yet_implemented(
            "ClinicalHistoryAgent", "agents/clinical_history_agent/agent.py"
        ) from exc

    agent = ClinicalHistoryAgent()
    result = agent.run(dict(state))

    logger.info(
        "[clinical_node] Done | clinical_fit_score=%.4f",
        result.get("clinical_analysis", {}).get("clinical_fit_score", -1),
    )
    return result


def rag_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: RAG Literature Agent
    Retrieves relevant PubMed papers for the detected finding.

    Reads:  state["vision_findings"], state["patient_data"]
    Writes: state["literature_results"]  (list of paper dicts with citations)
    """
    logger.info("[rag_node] Starting RAGLiteratureAgent")

    try:
        from agents.rag_literature_agent.agent import RAGLiteratureAgent  # noqa: PLC0415
    except (ImportError, AttributeError) as exc:
        raise _not_yet_implemented(
            "RAGLiteratureAgent", "agents/rag_literature_agent/agent.py"
        ) from exc

    agent = RAGLiteratureAgent()
    result = agent.run(dict(state))

    n_papers = len(result.get("literature_results", []))
    logger.info("[rag_node] Done | papers_retrieved=%d", n_papers)
    return result


def report_node(state: NeuroAgentState) -> dict[str, Any]:
    """
    Node: Report Generation Agent (fan-in point)
    Runs AFTER vision, clinical, and RAG all complete.
    Combines their outputs into a structured radiology report.

    Reads:  state["vision_findings"], state["clinical_analysis"],
            state["literature_results"], state["patient_data"]
    Writes: state["report"], state["overall_confidence"]
    """
    logger.info("[report_node] Starting ReportAgent (all three parallel nodes complete)")

    try:
        from agents.report_generation_agent.agent import ReportAgent  # noqa: PLC0415
    except (ImportError, AttributeError) as exc:
        raise _not_yet_implemented(
            "ReportAgent", "agents/report_generation_agent/agent.py"
        ) from exc

    agent = ReportAgent()
    result = agent.run(dict(state))

    logger.info(
        "[report_node] Done | overall_confidence=%.4f",
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
