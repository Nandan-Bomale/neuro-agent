"""
agent.py  —  Report Generation Agent
──────────────────────────────────────
The ReportGenerationAgent class is the clean public interface that the
LangGraph orchestrator calls. It wires together:

    LLMLoader (4-bit quantized Phi-3-mini or Llama)
        +
    PromptBuilder (assembles upstream agent outputs into a prompt)
        +
    Output parser (splits the LLM's text into structured fields)
        ↓
    ReportResult  (typed dataclass the orchestrator stores in graph state)

Compatibility note
──────────────────
The Clinical History Agent (agents/clinical_history_agent/agent.py) was
built independently and returns a standard dict. The run() method here
accepts EITHER a ClinicalOutput dataclass OR a raw dict — it converts
automatically. This keeps the orchestrator simple.

Usage
─────
    from agents.report_generation_agent.agent import ReportGenerationAgent
    from agents.report_generation_agent.prompt_templates import (
        PatientData, VisionOutput, RAGOutput,
    )

    agent = ReportGenerationAgent()           # uses Phi-3-mini by default
    result = agent.run(
        vision_output=vision_result,          # VisionOutput dataclass
        rag_output=rag_result,                # RAGOutput dataclass
        clinical_output=clinical_agent_dict,  # dict OR ClinicalOutput
        patient=patient_data,                 # PatientData dataclass
    )

    print(result.impression)
    print(result.confidence_level)            # "HIGH" | "MODERATE" | "LOW"
    print(result.requires_human_review)       # True if confidence < HIGH or parse errors
    print(result.full_report_text())          # full formatted report string
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Union

from agents.report_generation_agent.llm_loader import LLMLoader, DEFAULT_MODEL
from agents.report_generation_agent.prompt_templates import (
    PatientData,
    VisionOutput,
    RAGOutput,
    RAGCitation,
    ClinicalOutput,
    PromptBuilder,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass — what the orchestrator receives
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReportResult:
    """
    Structured output from the Report Generation Agent.

    All five report sections are parsed into individual fields so downstream
    agents (Verification Agent, Orchestrator) can inspect them without
    re-parsing raw text.
    """
    # ── Parsed report sections ────────────────────────────────────────────────
    findings: str
    impression: str
    supporting_evidence: str
    confidence_level: str           # "HIGH" | "MODERATE" | "LOW"
    confidence_score: float         # numeric value from Vision Agent (0.0–1.0)
    confidence_rationale: str
    recommendation: str

    # ── Metadata ──────────────────────────────────────────────────────────────
    patient_id: str
    model_used: str
    generation_time_s: float
    raw_output: str                 # full LLM output, preserved for debugging
    parse_warnings: list[str] = field(default_factory=list)

    # ── Verification flag ─────────────────────────────────────────────────────
    requires_human_review: bool = False
    """
    True when:
      - confidence_level is "LOW" or "MODERATE", OR
      - the parser could not cleanly extract one or more sections.
    The Verification Agent reads this flag to decide whether to route the
    case to mandatory human review.
    """

    def full_report_text(self) -> str:
        """
        Reconstruct the formatted report as a single string.
        Useful for display in the Streamlit frontend.
        """
        review_banner = (
            "\n⚠️  THIS CASE HAS BEEN FLAGGED FOR MANDATORY RADIOLOGIST REVIEW.\n"
            if self.requires_human_review else ""
        )
        return (
            f"{review_banner}"
            f"## FINDINGS\n{self.findings}\n\n"
            f"## IMPRESSION\n{self.impression}\n\n"
            f"## SUPPORTING EVIDENCE\n{self.supporting_evidence}\n\n"
            f"## CONFIDENCE\n"
            f"Level: {self.confidence_level}\n"
            f"Score: {self.confidence_score:.2f}\n"
            f"Rationale: {self.confidence_rationale}\n\n"
            f"## RECOMMENDATION\n{self.recommendation}"
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict for FastAPI response / JSON logging."""
        return {
            "patient_id": self.patient_id,
            "findings": self.findings,
            "impression": self.impression,
            "supporting_evidence": self.supporting_evidence,
            "confidence": {
                "level": self.confidence_level,
                "score": self.confidence_score,
                "rationale": self.confidence_rationale,
            },
            "recommendation": self.recommendation,
            "requires_human_review": self.requires_human_review,
            "model_used": self.model_used,
            "generation_time_s": round(self.generation_time_s, 2),
            "parse_warnings": self.parse_warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Report Generation Agent
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerationAgent:
    """
    Orchestrator-facing agent that generates structured radiology reports.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or path to fine-tuned local weights.
        Defaults to Phi-3-mini for laptop inference.
    hf_token : str | None
        HuggingFace token for gated models. Falls back to HF_TOKEN env var.
    max_new_tokens : int
        Maximum tokens the LLM generates. 600 covers all five sections
        comfortably. Increase to 800 if outputs are being cut off.
    temperature : float
        Low temperature keeps clinical reports factual and consistent.
        Range 0.1–0.3 recommended for medical text generation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        hf_token: Optional[str] = None,
        max_new_tokens: int = 600,
        temperature: float = 0.15,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._loader = LLMLoader(
            model_name=model_name,
            hf_token=hf_token,
        )
        self._builder = PromptBuilder()

        logger.info(
            "ReportGenerationAgent initialised (model=%s, max_new_tokens=%d)",
            model_name,
            max_new_tokens,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Explicitly load the LLM into VRAM.
        Call this at startup so the first report generation doesn't have
        a cold-start delay (~30s on RTX 3050).
        """
        self._loader.load()

    def run(
        self,
        vision_output: VisionOutput,
        rag_output: RAGOutput,
        clinical_output: Union[ClinicalOutput, dict],
        patient: PatientData,
    ) -> ReportResult:
        """
        Generate a structured radiology report.

        Parameters
        ----------
        vision_output   : VisionOutput from the Vision Agent
        rag_output      : RAGOutput from the RAG Literature Agent
        clinical_output : ClinicalOutput dataclass OR the raw dict returned
                          by ClinicalHistoryAgent.run() — both accepted.
        patient         : PatientData from the orchestrator

        Returns
        -------
        ReportResult — fully parsed, structured report ready for the
        Verification Agent and Streamlit frontend.
        """
        # Normalise clinical_output → ClinicalOutput dataclass
        clinical = self._normalise_clinical_output(clinical_output)

        logger.info(
            "ReportGenerationAgent.run() — patient=%s, finding=%s, confidence=%.2f",
            patient.patient_id,
            vision_output.finding,
            vision_output.confidence,
        )

        # 1. Build the prompt
        system_prompt, user_message = self._builder.build_report_prompt(
            vision_output=vision_output,
            rag_output=rag_output,
            clinical_output=clinical,
            patient=patient,
        )

        # 2. Pre-flight: check prompt fits in the context window
        length_check = self._builder.check_prompt_length(
            system_prompt,
            user_message,
            context_length=self._loader.context_length,
            max_new_tokens=self.max_new_tokens,
        )
        if not length_check["ok"]:
            logger.warning(
                "Prompt exceeds context window by ~%d tokens. "
                "Trimming RAG citations to top-2.",
                length_check["overflow_by"],
            )
            rag_output = self._trim_citations(rag_output, max_citations=2)
            system_prompt, user_message = self._builder.build_report_prompt(
                vision_output, rag_output, clinical, patient
            )

        # 3. Ensure model is loaded (no-op if already in VRAM)
        if not self._loader.is_loaded:
            self._loader.load()

        # 4. Generate
        t0 = time.perf_counter()
        raw_output = self._loader.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            repetition_penalty=1.1,
        )
        generation_time = time.perf_counter() - t0

        logger.info(
            "Generation complete in %.1fs — %d chars output.",
            generation_time,
            len(raw_output),
        )

        # 5. Parse LLM output into structured fields
        result = self._parse_output(
            raw_output=raw_output,
            vision_output=vision_output,
            patient=patient,
            generation_time=generation_time,
        )

        logger.info(
            "Report result — confidence=%s (%.2f), human_review=%s",
            result.confidence_level,
            result.confidence_score,
            result.requires_human_review,
        )
        return result

    def unload(self) -> None:
        """Free VRAM — call when switching to another GPU-heavy agent."""
        self._loader.unload()

    # ── Output parser ─────────────────────────────────────────────────────────

    def _parse_output(
        self,
        raw_output: str,
        vision_output: VisionOutput,
        patient: PatientData,
        generation_time: float,
    ) -> ReportResult:
        """
        Parse the five ## SECTION blocks out of the LLM's raw text output.

        Strategy
        ────────
        Split on `## SECTION_NAME` headers. If a section is missing, fill
        with a placeholder and log a parse warning — the Verification Agent
        will see requires_human_review=True in that case.
        """
        parse_warnings: list[str] = []

        sections = self._split_sections(raw_output)

        findings            = self._extract("FINDINGS",            sections, parse_warnings)
        impression          = self._extract("IMPRESSION",          sections, parse_warnings)
        supporting_evidence = self._extract("SUPPORTING EVIDENCE", sections, parse_warnings)
        confidence_block    = self._extract("CONFIDENCE",          sections, parse_warnings)
        recommendation      = self._extract("RECOMMENDATION",      sections, parse_warnings)

        # Parse the CONFIDENCE block into sub-fields
        conf_level, conf_rationale = self._parse_confidence_block(
            confidence_block, vision_output.confidence, parse_warnings
        )

        # Human review required for MODERATE/LOW confidence or parse failures
        requires_review = (
            conf_level in ("LOW", "MODERATE") or len(parse_warnings) > 0
        )

        return ReportResult(
            findings=findings,
            impression=impression,
            supporting_evidence=supporting_evidence,
            confidence_level=conf_level,
            confidence_score=vision_output.confidence,
            confidence_rationale=conf_rationale,
            recommendation=recommendation,
            patient_id=patient.patient_id,
            model_used=self.model_name,
            generation_time_s=generation_time,
            raw_output=raw_output,
            parse_warnings=parse_warnings,
            requires_human_review=requires_review,
        )

    @staticmethod
    def _split_sections(text: str) -> dict[str, str]:
        """
        Split the LLM output on `## SECTION_NAME` headers.

        Returns a dict mapping uppercase section name → content text.
        Also handles `**SECTION**` bold headers that smaller LLMs sometimes
        produce instead of markdown headings.
        """
        # Normalise bold-style headers → markdown headers
        text = re.sub(
            r"\*\*(FINDINGS|IMPRESSION|SUPPORTING EVIDENCE|"
            r"CONFIDENCE|RECOMMENDATION)\*\*",
            r"## \1",
            text,
        )

        pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
        matches = list(pattern.finditer(text))

        sections: dict[str, str] = {}
        for i, match in enumerate(matches):
            header = match.group(1).strip().upper()
            start  = match.end()
            end    = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections[header] = text[start:end].strip()

        return sections

    @staticmethod
    def _extract(
        section_name: str,
        sections: dict[str, str],
        warnings: list[str],
    ) -> str:
        """
        Extract a section by name with partial-match fallback.
        Logs a warning and returns a placeholder if the section is missing.
        """
        if section_name in sections:
            return sections[section_name].strip()

        # Partial match (e.g. "EVIDENCE" matches "SUPPORTING EVIDENCE")
        for key, value in sections.items():
            if section_name in key or key in section_name:
                return value.strip()

        warning = f"Section '{section_name}' not found in LLM output."
        warnings.append(warning)
        logger.warning(warning)
        return "[Section not generated — see raw_output for details]"

    @staticmethod
    def _parse_confidence_block(
        block: str,
        vision_score: float,
        warnings: list[str],
    ) -> tuple[str, str]:
        """
        Parse (level, rationale) from the CONFIDENCE section.

        Falls back to computing the label from the vision agent's numeric
        score if the LLM did not produce parseable output.
        """
        level     = None
        rationale = ""

        level_match = re.search(
            r"Level\s*:\s*(HIGH|MODERATE|LOW)", block, re.IGNORECASE
        )
        if level_match:
            level = level_match.group(1).upper()

        rationale_match = re.search(
            r"Rationale\s*:\s*(.+?)(?:\n|$)", block, re.IGNORECASE | re.DOTALL
        )
        if rationale_match:
            rationale = rationale_match.group(1).strip()

        # Compute level from numeric score if LLM didn't produce one
        if level is None:
            if vision_score >= 0.80:
                level = "HIGH"
            elif vision_score >= 0.55:
                level = "MODERATE"
            else:
                level = "LOW"
            warnings.append(
                f"Could not parse confidence level from output; "
                f"computed from vision score ({vision_score:.2f}) → {level}."
            )

        if not rationale:
            rationale = (
                f"Derived from vision model confidence score of {vision_score:.2f}."
            )

        return level, rationale

    # ── Compatibility helpers ─────────────────────────────────────────────────

    @staticmethod
    def _normalise_clinical_output(
        clinical: Union[ClinicalOutput, dict]
    ) -> ClinicalOutput:
        """
        Accept either a ClinicalOutput dataclass or the raw dict returned
        by the existing ClinicalHistoryAgent.run() implementation.

        The dict schema from ClinicalHistoryAgent.run():
            {
              "agent_name": "clinical_history_agent",
              "success": True,
              "output": {
                  "clinical_summary": str,
                  "consistency": "consistent" | "inconsistent" | "uncertain",
                  "key_factors": list[str],
                  "raw_reasoning": str,
              },
              "confidence": float,
              "error": None | str,
            }
        """
        if isinstance(clinical, ClinicalOutput):
            return clinical

        if isinstance(clinical, dict):
            output      = clinical.get("output", {})
            summary     = output.get("clinical_summary", "")
            consistency = output.get("consistency", "uncertain")
            key_factors = output.get("key_factors", [])

            return ClinicalOutput(
                clinical_reasoning=summary,
                risk_factors_identified=key_factors,
                history_consistent_with_finding=(consistency == "consistent"),
                notes=(
                    "History partially consistent with finding."
                    if consistency == "uncertain" else ""
                ),
            )

        raise TypeError(
            f"clinical_output must be ClinicalOutput or dict, "
            f"got {type(clinical)}"
        )

    @staticmethod
    def _trim_citations(rag_output: RAGOutput, max_citations: int) -> RAGOutput:
        """Keep only the top-N citations by relevance score."""
        trimmed = sorted(
            rag_output.citations,
            key=lambda c: c.relevance_score,
            reverse=True,
        )[:max_citations]
        return RAGOutput(query_used=rag_output.query_used, citations=trimmed)

    def __repr__(self) -> str:
        return (
            f"ReportGenerationAgent(model='{self.model_name}', "
            f"loaded={self._loader.is_loaded})"
        )
