"""
prompt_templates.py  (Clinical History Agent)
─────────────────────────────────────────────
This module re-exports the shared data-transfer objects and the
ClinicalHistory-specific prompt builder from the report agent's
prompt_templates module.

The Clinical History Agent uses the SAME underlying LLM (Phi-3-mini
or Llama-3.2-3B in 4-bit) as the Report Agent but with a completely
different system prompt (SYSTEM_PROMPT_CLINICAL).

Having this thin wrapper in the clinical_history_agent/ package means:
  - The orchestrator can import from here without knowing about report_agent
  - If the clinical agent ever needs its OWN specialised prompt logic, it
    can be added here without touching the report agent's module

Usage
─────
    from agents.clinical_history_agent.prompt_templates import (
        PromptBuilder,
        PatientData,
        VisionOutput,
        ClinicalOutput,
        SYSTEM_PROMPT_CLINICAL,
    )
"""

# Re-export everything needed from the shared prompt_templates module.
# The clinical agent uses the same dataclasses and PromptBuilder — it just
# always calls builder.build_clinical_prompt() instead of build_report_prompt().

from agents.report_generation_agent.prompt_templates import (  # noqa: F401
    SYSTEM_PROMPT_CLINICAL,
    PatientData,
    VisionOutput,
    ClinicalOutput,
    RAGOutput,
    RAGCitation,
    PromptBuilder,
)

__all__ = [
    "SYSTEM_PROMPT_CLINICAL",
    "PatientData",
    "VisionOutput",
    "ClinicalOutput",
    "RAGOutput",
    "RAGCitation",
    "PromptBuilder",
]
