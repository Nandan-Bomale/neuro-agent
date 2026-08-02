"""
patient_schema.py
-----------------
Pydantic models for structured patient input to the Clinical History Agent.

Design decisions
----------------
- Every field used in the LLM prompt is typed and validated here.
- Optional fields use None defaults — the prompt builder will handle missing
  values gracefully rather than failing.
- VisionFinding is a lightweight mirror of the scalar fields from
  VisionAgentResult.to_dict() — we only pull what we need for clinical
  reasoning; we do NOT pass the raw NumPy arrays across agent boundaries.
- Enums are used for sex and consistency so the orchestrator and downstream
  agents can rely on controlled vocabularies, not free strings.

Usage
-----
    from agents.clinical_history_agent.patient_schema import PatientData, VisionFinding

    patient = PatientData(
        patient_id="BraTS-001",
        age=62,
        sex="male",
        presenting_symptoms=["persistent headaches", "blurred vision"],
        symptom_duration_weeks=8,
        neurological_history=["hypertension"],
        current_medications=["amlodipine 5mg"],
    )

    finding = VisionFinding(
        tumour_detected=True,
        confidence_score=0.91,
        tumour_volume_cc=12.4,
        tumour_volume_voxels=12400,
        requires_review=False,
    )
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Controlled vocabulary enums
# ---------------------------------------------------------------------------

class Sex(str, Enum):
    """Biological sex — used to guide clinical reasoning (e.g. meningioma
    is more common in females; GBM is more common in males)."""
    MALE    = "male"
    FEMALE  = "female"
    OTHER   = "other"
    UNKNOWN = "unknown"


class ConsistencyLabel(str, Enum):
    """The three labels the Clinical History Agent can assign.

    - CONSISTENT:   The patient's clinical profile strongly supports the
                    vision finding (age, symptoms, history all align).
    - INCONSISTENT: There are notable clinical features that conflict with
                    the vision finding (e.g. no symptoms, young age, history
                    of a different pathology).
    - UNCERTAIN:    Evidence is mixed or insufficient to lean either way.
    """
    CONSISTENT   = "consistent"
    INCONSISTENT = "inconsistent"
    UNCERTAIN    = "uncertain"


# ---------------------------------------------------------------------------
# VisionFinding — what the Clinical History Agent receives from the Vision Agent
# ---------------------------------------------------------------------------

class VisionFinding(BaseModel):
    """Scalar subset of VisionAgentResult passed to the Clinical History Agent.

    Sourced from VisionAgentResult.to_dict() — NumPy arrays are excluded.

    Attributes:
        tumour_detected:      True if any voxel was classified as tumour.
        confidence_score:     Mean tumour-voxel sigmoid probability (0–1).
        tumour_volume_cc:     Estimated tumour volume in cubic centimetres.
        tumour_volume_voxels: Raw voxel count of the predicted tumour mask.
        requires_review:      True if confidence_score < 0.75 (REVIEW_THRESHOLD).
        gradcam_slice:        Axial slice index with highest Grad-CAM activation.
    """
    tumour_detected:      bool  = Field(
        ...,
        description="True if the Vision Agent detected a tumour region."
    )
    confidence_score:     float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Mean sigmoid probability over predicted tumour voxels (0–1)."
    )
    tumour_volume_cc:     float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated tumour volume in cubic centimetres (cc)."
    )
    tumour_volume_voxels: int   = Field(
        default=0,
        ge=0,
        description="Raw number of voxels classified as tumour."
    )
    requires_review:      bool  = Field(
        default=False,
        description="True if confidence_score < 0.75 — flagged for human review."
    )
    gradcam_slice:        Optional[int] = Field(
        default=None,
        description="Axial slice index selected for Grad-CAM overlay."
    )

    @classmethod
    def from_vision_dict(cls, d: dict) -> "VisionFinding":
        """Construct from VisionAgentResult.to_dict() output.

        Args:
            d: dict returned by VisionAgentResult.to_dict()

        Returns:
            VisionFinding instance.
        """
        return cls(
            tumour_detected      = d["tumour_detected"],
            confidence_score     = d["confidence_score"],
            tumour_volume_cc     = d.get("tumour_volume_cc", 0.0),
            tumour_volume_voxels = d.get("tumour_volume_voxels", 0),
            requires_review      = d.get("requires_review", False),
            gradcam_slice        = d.get("gradcam_slice"),
        )


# ---------------------------------------------------------------------------
# PatientData — the structured patient input
# ---------------------------------------------------------------------------

class PatientData(BaseModel):
    """Structured patient record for clinical history reasoning.

    All fields used in the LLM prompt are validated here.
    Optional fields degrade gracefully — the prompt builder notes when
    information is unavailable rather than fabricating it.

    Attributes:
        patient_id:               Unique identifier (BraTS case ID or synthetic ID).
        age:                      Patient age in years.
        sex:                      Biological sex (see Sex enum).
        presenting_symptoms:      List of symptom strings the patient presented with.
        symptom_duration_weeks:   How long symptoms have been present (weeks).
        neurological_history:     Prior neurological diagnoses or conditions.
        family_history:           Relevant family history of neurological disease.
        current_medications:      List of current medications (name + dose string).
        performance_status:       ECOG performance score 0–4 (0 = fully active).
        prior_imaging_findings:   Description of any previous MRI/CT findings.
        additional_notes:         Free-text clinical notes from the referring doctor.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    patient_id: str = Field(
        default="UNKNOWN",
        description="Unique patient / case identifier."
    )

    # ── Demographics ─────────────────────────────────────────────────────────
    age: int = Field(
        ...,
        ge=0, le=120,
        description="Patient age in years."
    )
    sex: Sex = Field(
        default=Sex.UNKNOWN,
        description="Biological sex (male / female / other / unknown)."
    )

    # ── Presenting complaint ──────────────────────────────────────────────────
    presenting_symptoms: List[str] = Field(
        default_factory=list,
        description=(
            "Symptoms the patient presented with, e.g. "
            "['persistent headaches', 'nausea', 'visual disturbances']."
        )
    )
    symptom_duration_weeks: Optional[int] = Field(
        default=None,
        ge=0,
        description="Duration of current symptoms in weeks. None if unknown."
    )

    # ── Medical history ───────────────────────────────────────────────────────
    neurological_history: List[str] = Field(
        default_factory=list,
        description=(
            "Prior neurological diagnoses or relevant conditions, "
            "e.g. ['epilepsy', 'hypertension']."
        )
    )
    family_history: List[str] = Field(
        default_factory=list,
        description=(
            "Relevant family history, "
            "e.g. ['father had glioblastoma', 'mother had meningioma']."
        )
    )

    # ── Medications ───────────────────────────────────────────────────────────
    current_medications: List[str] = Field(
        default_factory=list,
        description=(
            "Current medications with dose where known, "
            "e.g. ['dexamethasone 4mg', 'levetiracetam 500mg']."
        )
    )

    # ── Clinical context ──────────────────────────────────────────────────────
    performance_status: Optional[int] = Field(
        default=None,
        ge=0, le=4,
        description=(
            "ECOG performance status (0 = fully active, 4 = completely disabled). "
            "None if not assessed."
        )
    )
    prior_imaging_findings: Optional[str] = Field(
        default=None,
        description=(
            "Summary of prior MRI / CT / PET findings, if any. "
            "None if this is the patient's first imaging study."
        )
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description=(
            "Free-text clinical notes from the referring clinician. "
            "Used verbatim in the prompt if provided."
        )
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("presenting_symptoms", "neurological_history",
                     "family_history", "current_medications", mode="before")
    @classmethod
    def strip_empty_strings(cls, v: list) -> list:
        """Remove blank / whitespace-only entries from list fields."""
        if isinstance(v, list):
            return [item.strip() for item in v if str(item).strip()]
        return v

    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v) -> int:
        """Accept string ages like '62' gracefully."""
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError(f"age must be an integer, got: {v!r}")

    # ── Convenience methods ───────────────────────────────────────────────────

    def has_neurological_symptoms(self) -> bool:
        """Return True if any presenting symptoms are neurological in nature.

        Uses a simple keyword check — sufficient for prompt enrichment,
        not a clinical classifier.
        """
        neuro_keywords = {
            "headache", "seizure", "vision", "speech", "weakness",
            "numbness", "dizziness", "vertigo", "confusion", "memory",
            "balance", "coordination", "nausea", "vomiting",
        }
        all_symptoms = " ".join(self.presenting_symptoms).lower()
        return any(kw in all_symptoms for kw in neuro_keywords)

    def symptom_summary(self) -> str:
        """Return a compact symptom string for prompt insertion.

        Returns:
            Comma-joined symptoms, or 'no symptoms reported' if list is empty.
        """
        if not self.presenting_symptoms:
            return "no symptoms reported"
        return ", ".join(self.presenting_symptoms)

    def medication_summary(self) -> str:
        """Return a compact medication string for prompt insertion."""
        if not self.current_medications:
            return "none reported"
        return ", ".join(self.current_medications)

    def history_summary(self) -> str:
        """Return a compact neurological history string for prompt insertion."""
        if not self.neurological_history:
            return "no prior neurological history"
        return ", ".join(self.neurological_history)

    def __repr__(self) -> str:
        return (
            f"PatientData(id={self.patient_id!r}, age={self.age}, "
            f"sex={self.sex.value}, symptoms={self.presenting_symptoms})"
        )


# ---------------------------------------------------------------------------
# ClinicalHistoryInput — the combined input to the agent
# ---------------------------------------------------------------------------

class ClinicalHistoryInput(BaseModel):
    """Top-level input bundle passed to ClinicalHistoryAgent.run().

    Combines patient demographics + Vision Agent finding into one object
    so the orchestrator only needs to pass a single argument.

    Attributes:
        patient:       Structured patient record.
        vision_finding: Scalar output from the Vision Agent.
    """
    patient:        PatientData
    vision_finding: VisionFinding

    @classmethod
    def from_dicts(
        cls,
        patient_dict: dict,
        vision_dict:  dict,
    ) -> "ClinicalHistoryInput":
        """Convenience constructor from raw dicts (e.g. from JSON / API).

        Args:
            patient_dict: Dict matching PatientData fields.
            vision_dict:  Dict from VisionAgentResult.to_dict().

        Returns:
            ClinicalHistoryInput instance.
        """
        return cls(
            patient        = PatientData(**patient_dict),
            vision_finding = VisionFinding.from_vision_dict(vision_dict),
        )
