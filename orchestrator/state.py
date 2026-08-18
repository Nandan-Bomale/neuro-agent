"""
NeuroAgent — Shared Pipeline State
====================================
Single TypedDict that flows through every LangGraph node.
All agents READ from this dict and WRITE partial updates back into it.
The orchestrator (graph.py) merges each agent's output dict into this state.

Field ownership:
  Orchestrator    → run_id, timestamp, pipeline_status, error_message
  Vision Agent    → vision_findings
  Tumor Class.    → tumor_classification_findings
  Clinical Agent  → clinical_analysis
  RAG Agent       → literature_results
  Report Agent    → report, overall_confidence
  Verification    → verification_status, requires_human_review,
                    verification_notes, confidence_threshold
  Explainability  → gradcam_heatmap_path, explanation_summary
  Human Review    → human_review_reason  (set by human_review_node, no agent)

Usage:
    from orchestrator.state import NeuroAgentState, create_initial_state

    initial = create_initial_state(
        mri_scan_path="data/raw/patient_001.nii.gz",
        patient_data={"age": 45, "symptoms": ["headache"], "history": "..."},
        run_id="run-abc123",
    )
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from typing_extensions import TypedDict

# ── Pipeline status literals ───────────────────────────────────────────────────

PipelineStatus = Literal["running", "complete", "human_review_required", "error"]
VerificationStatus = Literal["pass", "human_review"]

# ── Shared state ───────────────────────────────────────────────────────────────


class NeuroAgentState(TypedDict, total=False):
    """
    Shared state dict that travels through every node in the LangGraph graph.

    All fields are optional (total=False) because the state is built up
    incrementally as each agent writes its output. The orchestrator starts
    the pipeline with only the input fields filled in; each agent adds its
    own keys and returns the updated dict.

    Agents must NOT delete or overwrite fields they don't own.
    """

    # ── Pipeline inputs (must be set before graph.invoke()) ───────────────────
    mri_scan_path: str
    """Absolute or relative path to the NIfTI MRI scan file."""

    mri_slice_path: Optional[str]
    """Path to the extracted 2D slice for Tumor Classification Agent."""

    patient_data: dict[str, Any]
    """
    Structured patient metadata dict. Expected keys (non-exhaustive):
      - age: int
      - sex: str            ("M" | "F" | "Other")
      - symptoms: list[str]
      - medical_history: str
      - medications: list[str]
      - referring_notes: str
    """

    # ── Pipeline metadata (set by orchestrator / nodes) ───────────────────────
    run_id: str
    """Unique identifier for this pipeline run (UUID4 string)."""

    timestamp: str
    """ISO 8601 UTC timestamp when the pipeline was started."""

    pipeline_status: PipelineStatus
    """Current lifecycle status of the pipeline run."""

    error_message: Optional[str]
    """Human-readable error description if pipeline_status == 'error'."""

    # ── Vision Agent output ───────────────────────────────────────────────────
    vision_findings: dict[str, Any]
    """
    Output from VisionAgent.run(). Expected keys:
      - confidence: float            — primary confidence score [0.0, 1.0]
      - detected_regions: list[dict] — bounding boxes / contours per region
      - segmentation_mask_path: str  — path to saved binary mask image
      - prediction_label: str        — e.g. "tumour_detected" | "no_tumour"
      - model_version: str           — e.g. "unet-monai-v1"
    """

    # ── Tumor Classification Agent output ──────────────────────────────────────
    tumor_classification_findings: dict[str, Any]
    """
    Output from TumorClassificationAgent.run(). Expected keys:
      - tumor_type: str             — "glioma" | "meningioma" | "pituitary" | "no_tumor"
      - tumor_grade: str | None     — "grade_II" | "grade_III" | "grade_IV" | None
      - type_probabilities: dict    — {type: float} softmax probs from ensemble
      - grade_probabilities: dict   — {grade: float} probs, empty if not glioma
      - clinical_urgency: str       — "urgent" | "monitor" | "routine" | "normal"
      - confidence: float           — geometric-mean confidence from TTA
      - tta_used: bool              — True if Test-Time Augmentation was applied
    """

    # ── Radiogenomics Agent output ────────────────────────────────────────────
    radiogenomics_findings: dict[str, Any]
    """
    Output from RadiogenomicsAgent.run(). Expected keys:
      - idh_mutation_status: str    — "mutant" | "wildtype"
      - mgmt_methylation_status: str — "methylated" | "unmethylated"
    """

    # ── Surgical Planning Agent output ────────────────────────────────────────
    surgical_analysis: dict[str, Any]
    """
    Output from SurgicalAgent.run(). Expected keys:
      - resectability_score: float  — e.g., 0.85
      - eloquent_area_proximity: str — "high" | "low"
    """

    # ── Prognostic Agent output ───────────────────────────────────────────────
    prognostic_analysis: dict[str, Any]
    """
    Output from PrognosticAgent.run(). Expected keys:
      - overall_survival_months: float
      - progression_free_survival_months: float
    """

    # ── Clinical Trial Agent output ───────────────────────────────────────────
    clinical_trials: list[dict[str, Any]]
    """
    Output from ClinicalTrialAgent.run(). List of relevant trials from clinicaltrials.gov.
    """

    # ── Neuro-Oncologist Agent output ─────────────────────────────────────────
    neuro_oncologist_plan: dict[str, Any]
    """
    Output from NeuroOncologistAgent.run(). The final treatment plan based on NCCN guidelines.
    Expected keys:
      - treatment_recommendation: str
      - chemotherapy_protocol: str
      - radiotherapy_protocol: str
      - generated_at: str
    """

    overall_confidence: float
    """
    Aggregated confidence score produced by the Report Agent [0.0, 1.0].
    Combines vision confidence + clinical fit score.
    Used by VerificationAgent as the primary signal.
    Falls back to vision_findings['confidence'] if not set.
    """

    # ── Verification Agent output (see agents/verification_agent/agent.py) ────
    verification_status: VerificationStatus
    """'pass' → continue to Explainability. 'human_review' → flag for review."""

    requires_human_review: bool
    """True if verification_status == 'human_review'."""

    verification_notes: str
    """Human-readable explanation of the verification decision."""

    confidence_threshold: float
    """The threshold value that was used during verification (audit trail)."""

    # ── Explainability Agent output ───────────────────────────────────────────
    gradcam_heatmap_path: str
    """Path to the saved Grad-CAM overlay image."""

    explanation_summary: str
    """Plain-language explanation of which scan regions drove the prediction."""

    # ── Human Review node output (no agent — set by human_review_node) ────────
    human_review_reason: str
    """Reason the case was flagged, forwarded from verification_notes."""


# ── Factory ────────────────────────────────────────────────────────────────────


def create_initial_state(
    mri_scan_path: str,
    patient_data: dict[str, Any],
    run_id: str | None = None,
) -> NeuroAgentState:
    """
    Build the starting state dict to pass into graph.invoke().

    Only populates the pipeline inputs + metadata. All agent output fields
    start absent (not None) — LangGraph merges them in as agents run.

    Args:
        mri_scan_path: Path to the NIfTI MRI file.
        patient_data:  Dict of patient metadata (age, symptoms, history, etc.).
        run_id:        Optional run identifier. Auto-generated UUID4 if omitted.

    Returns:
        A NeuroAgentState dict ready to be passed to compile_graph().invoke().

    Example:
        state = create_initial_state(
            mri_scan_path="data/raw/patient_001.nii.gz",
            patient_data={"age": 45, "symptoms": ["headache"]},
        )
        result = pipeline.invoke(state)
    """
    state: NeuroAgentState = {
        "mri_scan_path": mri_scan_path,
        "patient_data": patient_data,
        "run_id": run_id or str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_status": "running",
        "error_message": None,
    }
    return state
