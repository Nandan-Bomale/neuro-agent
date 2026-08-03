"""
Clinical History Agent
----------------------
Public interface for the NeuroAgent Clinical History module.

The Orchestrator imports from here:

    from agents.clinical_history_agent import ClinicalHistoryAgent

Everything else (schema, prompt builder, parser) is an implementation detail.
"""

from agents.clinical_history_agent.agent import ClinicalHistoryAgent
from agents.clinical_history_agent.patient_schema import (
    ClinicalHistoryInput,
    ConsistencyLabel,
    PatientData,
    Sex,
    VisionFinding,
)
from agents.clinical_history_agent.prompt_builder import PromptBuilder

__all__ = [
    "ClinicalHistoryAgent",
    "ClinicalHistoryInput",
    "ConsistencyLabel",
    "PatientData",
    "Sex",
    "VisionFinding",
    "PromptBuilder",
]
