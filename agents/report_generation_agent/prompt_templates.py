"""
prompt_templates.py
───────────────────
Prompt engineering for the NeuroAgent Report Generation Agent.

This module contains:
  1. SYSTEM_PROMPT_REPORT     — the system-role instruction for the LLM
  2. FEW_SHOT_EXAMPLE         — one complete worked example embedded in the prompt
  3. REPORT_OUTPUT_SCHEMA     — the exact section structure the LLM must follow
  4. PromptBuilder            — assembles inputs from upstream agents into a
                                 formatted user message ready for the LLM

Report output structure (5 mandatory sections)
───────────────────────────────────────────────
  ## FINDINGS
  ## IMPRESSION
  ## SUPPORTING EVIDENCE
  ## CONFIDENCE
  ## RECOMMENDATION

Input sources
─────────────
  • Vision Agent     → finding label, confidence score, tumour location/size
  • Clinical History Agent → clinical reasoning summary, risk factors
  • RAG Agent        → list of cited papers (title, PMID, relevance score)
  • Patient data     → age, sex, presenting symptoms, relevant history

Usage
─────
    from agents.report_generation_agent.prompt_templates import PromptBuilder

    builder = PromptBuilder()
    system_prompt, user_message = builder.build_report_prompt(
        vision_output=vision_result,
        rag_output=rag_result,
        clinical_output=clinical_result,
        patient=patient_data,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Data-transfer objects (shared across agents)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatientData:
    """Structured patient metadata passed by the orchestrator."""
    age: int
    sex: str                        # "Male" | "Female" | "Other"
    presenting_symptoms: list[str]
    medical_history: list[str]
    medications: list[str] = field(default_factory=list)
    scan_date: str = "Unknown"
    patient_id: str = "ANON"        # anonymised for academic use


@dataclass
class VisionOutput:
    """Output from the Vision Agent (U-Net + MONAI)."""
    finding: str                    # e.g. "Glioma", "No abnormality detected"
    confidence: float               # 0.0 – 1.0
    location: str                   # e.g. "Right frontal lobe"
    size_cm: Optional[float] = None # largest dimension in cm; None if no finding
    characteristics: list[str] = field(default_factory=list)
    # e.g. ["heterogeneous signal", "mass effect", "midline shift"]
    gradcam_available: bool = False


@dataclass
class RAGCitation:
    """A single paper returned by the RAG Literature Agent."""
    title: str
    pmid: str                       # PubMed ID
    year: int
    relevance_score: float          # 0.0 – 1.0
    key_finding: str                # one-sentence summary of the paper's
                                    # finding relevant to this case


@dataclass
class RAGOutput:
    """Output from the RAG Literature Agent."""
    query_used: str
    citations: list[RAGCitation]


@dataclass
class ClinicalOutput:
    """Output from the Clinical History Agent."""
    clinical_reasoning: str         # LLM-generated paragraph
    risk_factors_identified: list[str]
    history_consistent_with_finding: bool
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Report output schema — the exact structure the LLM must reproduce
# ─────────────────────────────────────────────────────────────────────────────

REPORT_OUTPUT_SCHEMA = """\
## FINDINGS
[Describe what was detected, including location, size, and imaging characteristics.
 If no abnormality was detected, state that clearly.]

## IMPRESSION
[Provide the clinical interpretation. State the most likely diagnosis based on
 the imaging findings AND the patient's clinical context.]

## SUPPORTING EVIDENCE
[List the cited medical literature that supports the impression.
 Format each citation as shown in the example below.
 Include only citations provided to you — do NOT hallucinate references.]

## CONFIDENCE
Level: HIGH | MODERATE | LOW
Score: [numeric value between 0.0 and 1.0]
Rationale: [One sentence explaining what drives the confidence level.]

## RECOMMENDATION
[State the recommended next step: e.g., surgical consult, MRI with contrast,
 PET scan, stereotactic biopsy, 3-month follow-up MRI, or discharge with advice.]
"""

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — Report Generation Agent
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_REPORT = """\
You are a senior neuroradiology AI assistant embedded in the NeuroAgent clinical \
decision-support pipeline. Your role is to synthesise imaging findings, clinical \
history, and peer-reviewed medical literature into a structured radiology report \
that can be reviewed by a qualified radiologist.

RULES YOU MUST FOLLOW
─────────────────────
1. Always produce exactly the five report sections in this order:
   FINDINGS → IMPRESSION → SUPPORTING EVIDENCE → CONFIDENCE → RECOMMENDATION
2. Use precise, professional clinical language. Avoid vague or lay terms.
3. Ground every claim in the evidence provided to you. Do NOT invent details,
   diagnoses, or references that were not given to you.
4. For SUPPORTING EVIDENCE: cite only the papers provided by the RAG agent.
   Format each citation as:
     [1] Author(s) (Year). Title. PMID: XXXXXXXX. Relevance: X.XX.
         Key finding: <one sentence>.
5. For CONFIDENCE: map the vision model's numeric score to a label:
     ≥ 0.80 → HIGH | 0.55 – 0.79 → MODERATE | < 0.55 → LOW
6. If confidence is LOW or MODERATE, the RECOMMENDATION must include:
     "⚠ Flag for mandatory radiologist review."
7. If no abnormality was detected (confidence < 0.5 on any finding), state
   "No significant intracranial abnormality detected on this study" in FINDINGS
   and recommend appropriate follow-up based on symptoms.
8. Do NOT begin your response with any preamble. Start directly with "## FINDINGS".

REPORT TEMPLATE
───────────────
""" + REPORT_OUTPUT_SCHEMA + """
ONE-SHOT EXAMPLE
────────────────
Below is one complete example of the input you will receive and the exact output
you must produce. Study the format carefully — your output must match it.

--- EXAMPLE INPUT ---
Patient: 52-year-old Male. Presenting: progressive headache (3 weeks), left arm \
weakness. History: hypertension. Medications: amlodipine.

Vision Agent:
  Finding: Glioma (suspected high-grade)
  Confidence: 0.91
  Location: Right frontal lobe, subcortical
  Size: 3.8 cm
  Characteristics: heterogeneous T2 signal, ring enhancement pattern, perilesional oedema

Clinical History Agent:
  Clinical reasoning: The patient's progressive headache and focal motor deficit \
(left arm weakness) are consistent with a right hemispheric space-occupying lesion. \
The three-week progression and the patient's age (52) raise concern for a primary \
high-grade glioma or metastatic deposit. Hypertension is unlikely to account for \
this presentation.
  Risk factors: age > 50, progressive neurological deficit, mass effect on imaging
  History consistent with finding: Yes

RAG Citations:
  [1] Ostrom et al. (2021). CBTRUS Statistical Report: Primary Brain and Other CNS \
Tumors. Neuro-Oncology. PMID: 34608945. Relevance: 0.95.
      Key finding: Glioblastoma (GBM) is the most common malignant primary brain \
tumour in adults, with peak incidence in the sixth decade of life.
  [2] Stupp et al. (2005). Radiotherapy plus Concomitant and Adjuvant Temozolomide \
for Glioblastoma. NEJM. PMID: 15758009. Relevance: 0.88.
      Key finding: Combined radiotherapy and temozolomide significantly improves \
survival in newly diagnosed GBM compared with radiotherapy alone.
  [3] Wen et al. (2020). RANO 2.0 Criteria for High-Grade Gliomas. \
Neuro-Oncology. PMID: 32780794. Relevance: 0.81.
      Key finding: Updated RANO criteria standardise MRI-based response assessment \
for high-grade gliomas in clinical trials and practice.

--- EXAMPLE OUTPUT ---

## FINDINGS
A 3.8 cm heterogeneous mass is identified in the right frontal lobe at the \
subcortical level. The lesion demonstrates ring-enhancement pattern and \
perilesional vasogenic oedema. There is associated mass effect on the adjacent \
white matter. No midline shift is identified on the current study. These \
characteristics are highly suspicious for a high-grade primary glial neoplasm.

## IMPRESSION
The imaging features — ring-enhancing mass with perilesional oedema in a \
52-year-old male presenting with progressive headache and contralateral motor \
deficit — are most consistent with a high-grade glioma, World Health \
Organisation (WHO) Grade IV (Glioblastoma) being the leading consideration. \
A metastatic deposit cannot be excluded without systemic staging; however, \
the solitary nature and frontal location are more typical of a primary tumour.

## SUPPORTING EVIDENCE
[1] Ostrom et al. (2021). CBTRUS Statistical Report: Primary Brain and Other CNS \
Tumors. Neuro-Oncology. PMID: 34608945. Relevance: 0.95.
    Key finding: Glioblastoma (GBM) is the most common malignant primary brain \
tumour in adults, with peak incidence in the sixth decade of life.

[2] Stupp et al. (2005). Radiotherapy plus Concomitant and Adjuvant Temozolomide \
for Glioblastoma. NEJM. PMID: 15758009. Relevance: 0.88.
    Key finding: Combined radiotherapy and temozolomide significantly improves \
survival in newly diagnosed GBM compared with radiotherapy alone.

[3] Wen et al. (2020). RANO 2.0 Criteria for High-Grade Gliomas. \
Neuro-Oncology. PMID: 32780794. Relevance: 0.81.
    Key finding: Updated RANO criteria standardise MRI-based response assessment \
for high-grade gliomas in clinical trials and practice.

## CONFIDENCE
Level: HIGH
Score: 0.91
Rationale: Strong alignment between the imaging phenotype (ring-enhancing \
heterogeneous mass), the patient's age and symptom trajectory, and the \
supporting literature on high-grade glioma presentation.

## RECOMMENDATION
Urgent neurosurgical consultation is recommended for consideration of \
stereotactic biopsy or surgical resection. Pre-operative MRI with gadolinium \
contrast and MR spectroscopy are advised to further characterise the lesion. \
Multidisciplinary team (MDT) review within 72 hours is indicated given the \
clinical urgency.

--- END OF EXAMPLE ---
"""


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — Clinical History Agent
# (same LLM, different persona and task)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_CLINICAL = """\
You are a clinical reasoning assistant embedded in the NeuroAgent pipeline. \
Your role is to analyse a patient's structured clinical record and reason about \
whether their age, symptoms, history, and medications are consistent with the \
finding reported by the vision agent.

You do NOT generate the final radiology report — that is handled by a separate \
agent. Your output is a structured clinical summary that will be passed as \
context to the Report Generation Agent.

OUTPUT FORMAT
─────────────
Produce your answer in exactly this structure:

CLINICAL REASONING:
[Two to four sentences. State whether the clinical picture is consistent with
 the imaging finding, and briefly explain why.]

RISK FACTORS IDENTIFIED:
- [factor 1]
- [factor 2]
- [factor 3, if applicable]

HISTORY CONSISTENT WITH FINDING: Yes | No | Partially

ADDITIONAL NOTES:
[Any relevant clinical caveat, differential to consider from history alone,
 or medication interaction relevant to MRI interpretation. Write "None" if
 there is nothing to add.]

RULES
─────
1. Do not speculate beyond the data provided.
2. Do not repeat the vision agent finding verbatim — add clinical interpretation.
3. Keep the output concise — it feeds into a larger prompt, not a standalone report.
4. Start directly with "CLINICAL REASONING:" — no preamble.

ONE-SHOT EXAMPLE
────────────────
--- EXAMPLE INPUT ---
Patient: 52-year-old Male.
Presenting symptoms: progressive headache (3 weeks), left arm weakness.
Medical history: hypertension (10 years).
Medications: amlodipine 5 mg once daily.
Vision Agent finding: Glioma (suspected high-grade), right frontal lobe, 3.8 cm, confidence 0.91.

--- EXAMPLE OUTPUT ---
CLINICAL REASONING:
The presenting triad of progressive headache, focal motor deficit (left arm \
weakness), and a right-sided frontal mass on imaging is clinically consistent \
with a space-occupying lesion exerting pressure on the contralateral motor \
cortex or corticospinal tracts. The three-week progression of symptoms in a \
52-year-old male is characteristic of a rapidly growing primary brain tumour \
rather than a benign or slow-growing process. Hypertension in this context is \
an incidental finding and does not account for the neurological presentation.

RISK FACTORS IDENTIFIED:
- Age > 50 (peak incidence decade for GBM)
- Progressive focal neurological deficit
- Three-week symptom trajectory (rapid progression)
- Male sex (slight epidemiological predisposition for high-grade glioma)

HISTORY CONSISTENT WITH FINDING: Yes

ADDITIONAL NOTES:
Amlodipine (calcium channel blocker) has no known interaction with MRI contrast \
agents. No contraindications to gadolinium contrast from the provided history. \
Differential from history: hypertensive encephalopathy is excluded given the \
focal nature and imaging characteristics.

--- END OF EXAMPLE ---
"""


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder — assembles structured inputs into LLM-ready prompt strings
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Assembles inputs from upstream agents into formatted prompt strings
    ready for the LLM.

    All methods return a (system_prompt, user_message) tuple which maps
    directly to the LLMLoader.generate() interface.
    """

    # ── Report Generation Agent ───────────────────────────────────────────────

    def build_report_prompt(
        self,
        vision_output: VisionOutput,
        rag_output: RAGOutput,
        clinical_output: ClinicalOutput,
        patient: PatientData,
    ) -> tuple[str, str]:
        """
        Build the full prompt for the Report Generation Agent.

        Returns
        -------
        (system_prompt, user_message)
        """
        user_message = self._format_report_user_message(
            vision_output, rag_output, clinical_output, patient
        )
        return SYSTEM_PROMPT_REPORT, user_message

    def _format_report_user_message(
        self,
        vision: VisionOutput,
        rag: RAGOutput,
        clinical: ClinicalOutput,
        patient: PatientData,
    ) -> str:
        """
        Format all upstream agent outputs into a single structured user message.
        """
        # ── Patient block ────────────────────────────────────────────────────
        symptoms_str = "; ".join(patient.presenting_symptoms) or "Not reported"
        history_str  = "; ".join(patient.medical_history) or "No significant history"
        meds_str     = "; ".join(patient.medications) or "None"

        patient_block = (
            f"Patient ID: {patient.patient_id}\n"
            f"Scan Date:  {patient.scan_date}\n"
            f"Age / Sex:  {patient.age}-year-old {patient.sex}\n"
            f"Presenting symptoms: {symptoms_str}\n"
            f"Medical history:     {history_str}\n"
            f"Medications:         {meds_str}"
        )

        # ── Vision Agent block ───────────────────────────────────────────────
        confidence_label = self._confidence_label(vision.confidence)
        size_str = f"{vision.size_cm} cm" if vision.size_cm else "Not measured"
        chars_str = (
            "; ".join(vision.characteristics)
            if vision.characteristics
            else "Not specified"
        )
        gradcam_str = "Available" if vision.gradcam_available else "Not available"

        vision_block = (
            f"Finding:          {vision.finding}\n"
            f"Confidence score: {vision.confidence:.2f} ({confidence_label})\n"
            f"Location:         {vision.location}\n"
            f"Size:             {size_str}\n"
            f"Characteristics:  {chars_str}\n"
            f"Grad-CAM overlay: {gradcam_str}"
        )

        # ── Clinical History block ───────────────────────────────────────────
        risk_str = (
            "\n  - ".join([""] + clinical.risk_factors_identified)
            if clinical.risk_factors_identified
            else "None identified"
        )
        consistent_str = (
            "Yes" if clinical.history_consistent_with_finding else "No"
        )
        notes_str = clinical.notes or "None"

        clinical_block = (
            f"Clinical reasoning:\n  {clinical.clinical_reasoning}\n\n"
            f"Risk factors identified:{risk_str}\n\n"
            f"History consistent with finding: {consistent_str}\n"
            f"Additional notes: {notes_str}"
        )

        # ── RAG Citations block ──────────────────────────────────────────────
        if rag.citations:
            citations_lines = []
            for i, c in enumerate(rag.citations, start=1):
                citations_lines.append(
                    f"  [{i}] Author(s) ({c.year}). {c.title}. "
                    f"PMID: {c.pmid}. Relevance: {c.relevance_score:.2f}.\n"
                    f"      Key finding: {c.key_finding}"
                )
            citations_str = "\n".join(citations_lines)
        else:
            citations_str = "  No citations retrieved. Proceed based on imaging findings only."

        rag_block = (
            f"RAG query used: \"{rag.query_used}\"\n\n"
            f"Retrieved citations:\n{citations_str}"
        )

        # ── Assemble full user message ───────────────────────────────────────
        user_message = f"""\
Generate a structured radiology report for the following case. \
Follow the five-section format (FINDINGS, IMPRESSION, SUPPORTING EVIDENCE, \
CONFIDENCE, RECOMMENDATION) exactly as shown in your instructions.

━━━ PATIENT DATA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{patient_block}

━━━ VISION AGENT OUTPUT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{vision_block}

━━━ CLINICAL HISTORY AGENT OUTPUT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{clinical_block}

━━━ RAG LITERATURE AGENT OUTPUT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{rag_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Now write the structured radiology report. Start with ## FINDINGS.\
"""
        return user_message

    # ── Clinical History Agent ────────────────────────────────────────────────

    def build_clinical_prompt(
        self,
        patient: PatientData,
        vision_output: VisionOutput,
    ) -> tuple[str, str]:
        """
        Build the prompt for the Clinical History Agent.

        Returns
        -------
        (system_prompt, user_message)
        """
        user_message = self._format_clinical_user_message(patient, vision_output)
        return SYSTEM_PROMPT_CLINICAL, user_message

    def _format_clinical_user_message(
        self,
        patient: PatientData,
        vision: VisionOutput,
    ) -> str:
        symptoms_str = "; ".join(patient.presenting_symptoms) or "Not reported"
        history_str  = "; ".join(patient.medical_history) or "No significant history"
        meds_str     = "; ".join(patient.medications) or "None"

        return f"""\
Analyse the following patient record and imaging finding. \
Produce your structured clinical summary in the format specified.

Patient: {patient.age}-year-old {patient.sex}.
Presenting symptoms: {symptoms_str}.
Medical history: {history_str}.
Medications: {meds_str}.
Vision Agent finding: {vision.finding}, {vision.location}\
{f", {vision.size_cm} cm" if vision.size_cm else ""}, \
confidence {vision.confidence:.2f}.

Produce the clinical summary now. Start with "CLINICAL REASONING:"\
"""

    # ── Shared utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _confidence_label(score: float) -> str:
        """Map a numeric confidence score to a HIGH / MODERATE / LOW label."""
        if score >= 0.80:
            return "HIGH"
        elif score >= 0.55:
            return "MODERATE"
        else:
            return "LOW"

    @staticmethod
    def estimate_token_count(text: str) -> int:
        """
        Rough token estimate: ~4 characters per token (GPT-style rule of thumb).
        Used for pre-flight checks before sending to the LLM.
        """
        return len(text) // 4

    def check_prompt_length(
        self,
        system_prompt: str,
        user_message: str,
        context_length: int = 4096,
        max_new_tokens: int = 512,
    ) -> dict:
        """
        Estimate whether the assembled prompt fits within the model's context
        window, leaving room for max_new_tokens of generation.

        Returns a dict with keys: ok (bool), total_tokens, budget_tokens.
        """
        total = self.estimate_token_count(system_prompt + user_message)
        budget = context_length - max_new_tokens
        return {
            "ok": total <= budget,
            "estimated_input_tokens": total,
            "budget_tokens": budget,
            "overflow_by": max(0, total - budget),
        }
