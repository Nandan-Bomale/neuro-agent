"""
prompt_builder.py
-----------------
Constructs the structured LLM reasoning prompt for the Clinical History Agent.

Design decisions
----------------
- The prompt is split into a fixed SYSTEM prompt and a dynamic USER prompt.
  The system prompt tells Phi-3-mini its role and output format.
  The user prompt injects the actual patient data + Vision Agent finding.

- We use Phi-3-mini's native chat template format:
      <|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n
  This matches the tokenizer's apply_chat_template() output exactly, so the
  model produces well-formed responses without extra formatting tokens.

- The prompt asks the model to reason step-by-step (CoT) FIRST, then output
  a structured block. This produces better consistency judgements on Phi-3-mini
  than asking for the answer directly.

- The structured output block uses a simple KEY: value format (not JSON) because
  Phi-3-mini-4k is more reliably consistent with plain-text structured output
  than JSON, and the agent.py parser handles both formats anyway.

- All patient fields are safe-defaulted here — the builder never raises on
  missing data, it surfaces "not available" in the prompt instead.

Usage
-----
    from agents.clinical_history_agent.patient_schema import (
        PatientData, VisionFinding
    )
    from agents.clinical_history_agent.prompt_builder import PromptBuilder

    builder = PromptBuilder()

    patient = PatientData(age=62, sex="male",
                          presenting_symptoms=["headaches", "nausea"],
                          symptom_duration_weeks=8)

    finding = VisionFinding(tumour_detected=True,
                             confidence_score=0.91,
                             tumour_volume_cc=12.4,
                             requires_review=False)

    system_prompt, user_prompt = builder.build(patient, finding)
    full_prompt = builder.format_for_phi3(system_prompt, user_prompt)
"""

from __future__ import annotations

from textwrap import dedent
from typing import Tuple

from agents.clinical_history_agent.patient_schema import PatientData, VisionFinding


# ---------------------------------------------------------------------------
# System prompt — fixed, defines the model's role and output contract
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = dedent("""\
    You are a clinical reasoning assistant supporting a radiologist.
    Your task is to evaluate whether a patient's clinical profile is consistent
    with an abnormality detected in a brain MRI scan by an automated Vision System.

    You will be given:
    1. The patient's demographics, symptoms, medical history, and medications.
    2. The Vision System's finding: whether a tumour was detected, its estimated
       volume, and the system's confidence score.

    Your job is to reason carefully and then produce a structured clinical summary.

    Instructions:
    - Think step-by-step. Consider the patient's age, sex, symptom pattern,
      symptom duration, neurological history, family history, and medications.
    - Assess whether this clinical profile commonly presents with the detected
      finding. Consider relevant epidemiology (e.g. GBM peaks at 55–75 years,
      meningioma is more common in females, etc.).
    - Be concise but clinically grounded. Do not fabricate facts.
    - If information is missing, state that it is unavailable rather than
      making assumptions.
    - End your response with a structured block in EXACTLY this format:

    ---CLINICAL SUMMARY---
    SUMMARY: <1-3 sentence prose summary for the radiologist>
    CONSISTENCY: <consistent | inconsistent | uncertain>
    CONFIDENCE: <a float between 0.00 and 1.00 representing your certainty>
    KEY_FACTORS: <comma-separated list of the 2-4 most important clinical factors>
    ---END SUMMARY---

    Do not add any text after ---END SUMMARY---.\
""")


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Constructs structured LLM prompts for clinical history reasoning.

    All patient fields are formatted with safe defaults — missing values
    appear as 'not available' in the prompt rather than raising errors.

    The builder produces two strings:
      - system_prompt: fixed role definition (can be cached)
      - user_prompt:   dynamic patient + finding context (changes per patient)

    The pair is formatted for Phi-3-mini using format_for_phi3().

    Attributes:
        system_prompt: The fixed system-role string (SYSTEM_PROMPT constant).
    """

    def __init__(self) -> None:
        self.system_prompt: str = SYSTEM_PROMPT

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        patient: PatientData,
        finding: VisionFinding,
    ) -> Tuple[str, str]:
        """Build the system and user prompt strings for one patient case.

        Args:
            patient: Validated PatientData instance.
            finding: VisionFinding from VisionAgentResult.to_dict().

        Returns:
            Tuple[system_prompt, user_prompt] — both plain strings.
        """
        user_prompt = self._build_user_prompt(patient, finding)
        return self.system_prompt, user_prompt

    def format_for_phi3(self, system_prompt: str, user_prompt: str) -> str:
        """Format the system + user prompt into Phi-3-mini's chat template.

        Phi-3-mini uses special tokens:
            <|system|>\\n{content}<|end|>\\n
            <|user|>\\n{content}<|end|>\\n
            <|assistant|>\\n

        The final <|assistant|>\\n signals the model to begin generating.

        Args:
            system_prompt: The system role string.
            user_prompt:   The user turn string.

        Returns:
            Full prompt string ready for tokenizer.encode() / model.generate().
        """
        return (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{user_prompt}<|end|>\n"
            f"<|assistant|>\n"
        )

    def build_formatted(
        self,
        patient: PatientData,
        finding: VisionFinding,
    ) -> str:
        """Convenience: build + format in one call.

        Args:
            patient: Validated PatientData instance.
            finding: VisionFinding from the Vision Agent.

        Returns:
            Full Phi-3-mini formatted prompt string.
        """
        system, user = self.build(patient, finding)
        return self.format_for_phi3(system, user)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_user_prompt(
        self,
        patient: PatientData,
        finding: VisionFinding,
    ) -> str:
        """Assemble the dynamic user prompt from patient data + vision finding.

        Each section is built by a dedicated helper so it's easy to adjust
        individual parts without touching the overall structure.

        Args:
            patient: PatientData instance.
            finding: VisionFinding instance.

        Returns:
            User prompt string.
        """
        sections = [
            self._section_patient_demographics(patient),
            self._section_clinical_presentation(patient),
            self._section_medical_history(patient),
            self._section_vision_finding(finding),
            self._section_task_instruction(),
        ]
        return "\n\n".join(sections)

    # ── Section builders ──────────────────────────────────────────────────────

    @staticmethod
    def _section_patient_demographics(patient: PatientData) -> str:
        """Format patient demographics section."""
        age_str = str(patient.age)
        sex_str = patient.sex.value

        ps_str = "not assessed"
        if patient.performance_status is not None:
            ps_descriptions = {
                0: "0 — fully active, able to carry on all pre-disease activities",
                1: "1 — restricted in physically strenuous activity but ambulatory",
                2: "2 — ambulatory and capable of all self-care but unable to work",
                3: "3 — limited self-care, confined to bed >50% of waking hours",
                4: "4 — completely disabled, no self-care, confined to bed/chair",
            }
            ps_str = ps_descriptions.get(
                patient.performance_status,
                str(patient.performance_status)
            )

        return dedent(f"""\
            ## Patient Demographics
            - Patient ID : {patient.patient_id}
            - Age        : {age_str} years
            - Sex        : {sex_str}
            - ECOG PS    : {ps_str}\
        """)

    @staticmethod
    def _section_clinical_presentation(patient: PatientData) -> str:
        """Format presenting symptoms and duration section."""
        symptom_str = patient.symptom_summary()

        duration_str = "not available"
        if patient.symptom_duration_weeks is not None:
            w = patient.symptom_duration_weeks
            if w == 0:
                duration_str = "acute presentation (< 1 week)"
            elif w <= 4:
                duration_str = f"{w} week(s) — subacute"
            elif w <= 12:
                duration_str = f"{w} weeks — months"
            else:
                duration_str = f"{w} weeks (~{w // 4} months) — chronic"

        neuro_flag = (
            "Yes — patient has neurological symptoms consistent with CNS involvement."
            if patient.has_neurological_symptoms()
            else "No neurological symptoms detected in the symptom list."
        )

        return dedent(f"""\
            ## Clinical Presentation
            - Presenting symptoms    : {symptom_str}
            - Symptom duration       : {duration_str}
            - Neurological symptoms? : {neuro_flag}\
        """)

    @staticmethod
    def _section_medical_history(patient: PatientData) -> str:
        """Format medical history, medications, and family history section."""
        lines = [
            "## Medical History",
            f"- Neurological history  : {patient.history_summary()}",
            f"- Current medications   : {patient.medication_summary()}",
        ]

        if patient.family_history:
            lines.append(
                f"- Family history        : {', '.join(patient.family_history)}"
            )
        else:
            lines.append("- Family history        : none reported")

        if patient.prior_imaging_findings:
            lines.append(
                f"- Prior imaging         : {patient.prior_imaging_findings}"
            )
        else:
            lines.append("- Prior imaging         : none on record")

        if patient.additional_notes:
            lines.append(
                f"- Additional notes      : {patient.additional_notes}"
            )

        return "\n".join(lines)

    @staticmethod
    def _section_vision_finding(finding: VisionFinding) -> str:
        """Format Vision Agent finding section."""
        detected_str   = "TUMOUR DETECTED" if finding.tumour_detected else "NO TUMOUR DETECTED"
        confidence_pct = f"{finding.confidence_score * 100:.1f}%"
        volume_str     = (
            f"{finding.tumour_volume_cc:.1f} cc ({finding.tumour_volume_voxels} voxels)"
            if finding.tumour_detected
            else "N/A (no tumour detected)"
        )
        review_str = (
            "[REVIEW REQUIRED]  YES — confidence is below the review threshold (0.75). "
            "The Vision Agent is uncertain; treat this finding with caution."
            if finding.requires_review
            else "No — confidence is within the acceptable range."
        )

        confidence_note = _confidence_note(finding.confidence_score)

        return dedent(f"""\
            ## Vision Agent Finding
            - Detection result   : {detected_str}
            - Confidence score   : {finding.confidence_score:.3f} ({confidence_pct}) — {confidence_note}
            - Estimated volume   : {volume_str}
            - Requires review?   : {review_str}\
        """)

    @staticmethod
    def _section_task_instruction() -> str:
        """Final task instruction reminding the model of what to produce."""
        return dedent("""\
            ## Your Task
            Given the patient information and the Vision Agent's finding above:

            1. Reason step-by-step about whether this patient's clinical profile
               is consistent with the detected finding. Consider:
               - Does the patient's age and sex fit the typical demographic for
                 the detected abnormality?
               - Are the presenting symptoms and their duration consistent with
                 an intracranial mass or tumour?
               - Does the neurological history or medication list provide
                 additional supporting or conflicting evidence?
               - Does the volume and confidence of the vision finding align with
                 the clinical presentation?

            2. After your reasoning, produce the structured summary block
               in the exact format specified in your instructions.\
        """)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _confidence_note(score: float) -> str:
    """Return a human-readable confidence band description.

    Args:
        score: Float 0–1 confidence score.

    Returns:
        Short descriptive string for embedding in the prompt.
    """
    if score >= 0.90:
        return "high confidence"
    elif score >= 0.75:
        return "moderate-high confidence"
    elif score >= 0.60:
        return "moderate confidence — borderline"
    else:
        return "low confidence — result uncertain"
