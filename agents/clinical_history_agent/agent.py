"""
agent.py
--------
ClinicalHistoryAgent — the public interface for the Clinical History Agent.

This is the only file the Orchestrator needs to import from this module.

Pipeline position
-----------------
    Vision Agent
        │  (VisionAgentResult.to_dict())
        ▼
    ClinicalHistoryAgent   ← this file
        │  (standard agent output dict)
        ▼
    Report Generation Agent

Interface contract
------------------
    Input  : ClinicalHistoryInput (PatientData + VisionFinding)
             OR raw dicts via ClinicalHistoryAgent.run_from_dicts()
    Output : standard agent dict:
                {
                  "agent_name" : "clinical_history_agent",
                  "success"    : True,
                  "output"     : {
                      "clinical_summary"  : str,   # prose for the Report Agent
                      "consistency"       : str,   # consistent|inconsistent|uncertain
                      "key_factors"       : list,  # 2-4 deciding clinical factors
                      "raw_reasoning"     : str,   # full model reasoning (chain-of-thought)
                  },
                  "confidence" : float,            # model's stated confidence 0–1
                  "error"      : None | str,
                }

LLM used
--------
    microsoft/Phi-3-mini-4k-instruct  (4-bit NF4 via bitsandbytes)
    VRAM usage: ~2.5GB — fits comfortably on RTX 3050 (4GB)
    No fine-tuning — prompt-engineering only.

Usage
-----
    from agents.clinical_history_agent.agent import ClinicalHistoryAgent

    agent = ClinicalHistoryAgent()

    result = agent.run_from_dicts(
        patient_dict={
            "patient_id": "BraTS-001",
            "age": 62,
            "sex": "male",
            "presenting_symptoms": ["persistent headaches", "nausea"],
            "symptom_duration_weeks": 8,
            "neurological_history": ["hypertension"],
            "current_medications": ["dexamethasone 4mg"],
        },
        vision_dict={
            "tumour_detected":      True,
            "confidence_score":     0.91,
            "tumour_volume_cc":     12.4,
            "tumour_volume_voxels": 12400,
            "requires_review":      False,
        }
    )

    print(result["output"]["clinical_summary"])
    print(result["output"]["consistency"])
"""

from __future__ import annotations

import re
import time
import traceback
from typing import Any, Dict, Optional

import torch

from agents.clinical_history_agent.patient_schema import (
    ClinicalHistoryInput,
    ConsistencyLabel,
    PatientData,
    VisionFinding,
)
from agents.clinical_history_agent.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: HuggingFace model ID. Must match the tokenizer.
MODEL_ID: str = "microsoft/Phi-3-mini-4k-instruct"

#: Maximum new tokens to generate. ~400 covers CoT reasoning + summary block.
MAX_NEW_TOKENS: int = 512

#: Generation temperature — lower = more deterministic, better for structured output.
TEMPERATURE: float = 0.2

#: Top-p nucleus sampling.
TOP_P: float = 0.9

#: Default confidence when model output cannot be parsed.
FALLBACK_CONFIDENCE: float = 0.5

#: Minimum allowed model confidence (clamp lower bound).
MIN_CONFIDENCE: float = 0.0

#: Maximum allowed model confidence (clamp upper bound).
MAX_CONFIDENCE: float = 1.0


# ---------------------------------------------------------------------------
# ClinicalHistoryAgent
# ---------------------------------------------------------------------------

class ClinicalHistoryAgent:
    """Clinical History Agent for brain MRI diagnosis support.

    Loads Phi-3-mini-4k-instruct in 4-bit quantization and uses
    prompt engineering to reason about whether a patient's clinical profile
    is consistent with the Vision Agent's finding.

    The model is loaded lazily on first .run() call to avoid GPU memory
    allocation when the agent object is constructed at startup.

    Args:
        model_id:        HuggingFace model ID. Defaults to Phi-3-mini-4k-instruct.
        device:          torch.device. Auto-detected (CUDA preferred, CPU fallback).
        use_4bit:        If True, load with 4-bit NF4 quantization (bitsandbytes).
                         Requires CUDA. On CPU, ignored and FP32 is used.
        max_new_tokens:  Maximum tokens to generate per call.
        temperature:     Sampling temperature (lower = more deterministic).
        top_p:           Nucleus sampling top-p value.
    """

    def __init__(
        self,
        model_id:       str            = MODEL_ID,
        device:         Optional[torch.device] = None,
        use_4bit:       bool           = True,
        max_new_tokens: int            = MAX_NEW_TOKENS,
        temperature:    float          = TEMPERATURE,
        top_p:          float          = TOP_P,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_id       = model_id
        self.device         = device
        self.use_4bit       = use_4bit and (device.type == "cuda")
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_p          = top_p

        self._model     = None
        self._tokenizer = None
        self._pipeline  = None

        self._prompt_builder = PromptBuilder()

        print(
            f"[ClinicalHistoryAgent] Initialised | "
            f"model={model_id} | device={device} | 4bit={self.use_4bit}"
        )

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """Load Phi-3-mini and tokenizer on first call (lazy init).

        Uses 4-bit NF4 quantization via bitsandbytes when CUDA is available.
        Falls back to FP32 on CPU (slower, but functional for testing).
        """
        if self._pipeline is not None:
            return

        print(f"[ClinicalHistoryAgent] Loading {self.model_id} ...")
        t0 = time.time()

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit               = True,
                bnb_4bit_quant_type        = "nf4",
                bnb_4bit_compute_dtype     = torch.float16,
                bnb_4bit_use_double_quant  = True,   # extra ~0.3 bits saving
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config = bnb_config,
                device_map          = "auto",
                trust_remote_code   = True,
            )
        else:
            # CPU fallback — used for unit tests without GPU
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype       = torch.float32,
                trust_remote_code = True,
            )
            model = model.to(self.device)

        self._tokenizer = tokenizer
        self._model     = model

        # HuggingFace text-generation pipeline — handles tokenization + sampling
        self._pipeline = pipeline(
            task      = "text-generation",
            model     = self._model,
            tokenizer = self._tokenizer,
        )

        elapsed = time.time() - t0
        print(f"[ClinicalHistoryAgent] Model loaded in {elapsed:.1f}s")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, input_data: ClinicalHistoryInput) -> Dict[str, Any]:
        """Run clinical history reasoning for one patient case.

        Steps:
          1. Build the structured LLM prompt from patient + vision finding.
          2. Run Phi-3-mini inference.
          3. Parse the structured summary block from the model output.
          4. Return the standard agent output dict.

        Args:
            input_data: ClinicalHistoryInput (PatientData + VisionFinding).

        Returns:
            Standard agent output dict (see module docstring for schema).
        """
        t_start = time.time()

        try:
            self._ensure_model_loaded()

            # ── Step 1: Build prompt ─────────────────────────────────────────
            full_prompt = self._prompt_builder.build_formatted(
                patient = input_data.patient,
                finding = input_data.vision_finding,
            )

            print(
                f"[ClinicalHistoryAgent] Running inference | "
                f"patient={input_data.patient.patient_id} | "
                f"tumour_detected={input_data.vision_finding.tumour_detected}"
            )

            # ── Step 2: Generate ─────────────────────────────────────────────
            raw_output = self._generate(full_prompt)

            # ── Step 3: Parse ────────────────────────────────────────────────
            parsed = _parse_model_output(raw_output)

            elapsed = time.time() - t_start
            print(
                f"[ClinicalHistoryAgent] Done | "
                f"consistency={parsed['consistency']} | "
                f"confidence={parsed['confidence']:.2f} | "
                f"elapsed={elapsed:.1f}s"
            )

            return {
                "agent_name": "clinical_history_agent",
                "success":    True,
                "output": {
                    "clinical_summary": parsed["summary"],
                    "consistency":      parsed["consistency"],
                    "key_factors":      parsed["key_factors"],
                    "raw_reasoning":    raw_output,
                },
                "confidence": parsed["confidence"],
                "error":      None,
            }

        except Exception as exc:
            elapsed = time.time() - t_start
            error_msg = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            print(f"[ClinicalHistoryAgent] ERROR after {elapsed:.1f}s:\n{tb}")

            return {
                "agent_name": "clinical_history_agent",
                "success":    False,
                "output": {
                    "clinical_summary": "Clinical history reasoning failed. Manual review required.",
                    "consistency":      ConsistencyLabel.UNCERTAIN.value,
                    "key_factors":      [],
                    "raw_reasoning":    "",
                },
                "confidence": FALLBACK_CONFIDENCE,
                "error":      error_msg,
            }

    def run_from_dicts(
        self,
        patient_dict: dict,
        vision_dict:  dict,
    ) -> Dict[str, Any]:
        """Convenience method: build ClinicalHistoryInput from raw dicts and run.

        This is the primary entry point for the Orchestrator, which passes
        dicts from its LangGraph state rather than Pydantic objects.

        Args:
            patient_dict: Dict matching PatientData fields (from API / JSON).
            vision_dict:  Dict from VisionAgentResult.to_dict().

        Returns:
            Standard agent output dict.
        """
        input_data = ClinicalHistoryInput.from_dicts(
            patient_dict = patient_dict,
            vision_dict  = vision_dict,
        )
        return self.run(input_data)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        """Run text generation and return the assistant's response only.

        Strips the prompt prefix from the output — HuggingFace pipelines
        return the full prompt + generation by default.

        Args:
            prompt: Full formatted prompt string (Phi-3 chat template).

        Returns:
            The model's generated text only (no prompt prefix).
        """
        outputs = self._pipeline(
            prompt,
            max_new_tokens  = self.max_new_tokens,
            temperature     = self.temperature,
            top_p           = self.top_p,
            do_sample       = True,
            return_full_text= False,   # return only the new tokens, not the prompt
            pad_token_id    = self._tokenizer.eos_token_id,
        )
        # Pipeline returns list of dicts: [{"generated_text": "..."}]
        return outputs[0]["generated_text"].strip()

    def __repr__(self) -> str:
        loaded = self._pipeline is not None
        return (
            f"ClinicalHistoryAgent("
            f"model={self.model_id!r}, "
            f"device={self.device}, "
            f"4bit={self.use_4bit}, "
            f"loaded={loaded})"
        )


# ---------------------------------------------------------------------------
# Output parser — extracts the structured block from model output
# ---------------------------------------------------------------------------

def _parse_model_output(text: str) -> Dict[str, Any]:
    """Parse the structured summary block from model-generated text.

    Expected format (anywhere in the text):
        ---CLINICAL SUMMARY---
        SUMMARY: <text>
        CONSISTENCY: <consistent | inconsistent | uncertain>
        CONFIDENCE: <float>
        KEY_FACTORS: <comma-separated list>
        ---END SUMMARY---

    Design: regex-based extraction is more robust than JSON parsing for
    Phi-3-mini, which sometimes produces minor formatting deviations.
    All fields have safe fallbacks.

    Args:
        text: Raw model-generated string (chain-of-thought + summary block).

    Returns:
        Dict with keys: summary, consistency, confidence, key_factors.
    """
    result = {
        "summary":     "Clinical summary could not be extracted from model output.",
        "consistency": ConsistencyLabel.UNCERTAIN.value,
        "confidence":  FALLBACK_CONFIDENCE,
        "key_factors": [],
    }

    # ── Extract the block between the delimiters ──────────────────────────────
    block_match = re.search(
        r"---CLINICAL SUMMARY---(.*?)---END SUMMARY---",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not block_match:
        # Fallback: try to extract individual fields from anywhere in the text
        # (model sometimes drops the delimiters but keeps the KEY: value lines)
        _extract_fields_loose(text, result)
        return result

    block = block_match.group(1)
    _extract_fields_from_block(block, result)
    return result


def _extract_fields_from_block(block: str, result: Dict[str, Any]) -> None:
    """Extract key-value fields from the structured block.

    Modifies result dict in-place.

    Args:
        block:  Text between the delimiters.
        result: Dict to update with extracted values.
    """
    # SUMMARY
    summary_match = re.search(
        r"SUMMARY\s*:\s*(.+?)(?=\n[A-Z_]+\s*:|$)",
        block,
        re.DOTALL | re.IGNORECASE,
    )
    if summary_match:
        result["summary"] = summary_match.group(1).strip()

    # CONSISTENCY
    consistency_match = re.search(
        r"CONSISTENCY\s*:\s*(consistent|inconsistent|uncertain)",
        block,
        re.IGNORECASE,
    )
    if consistency_match:
        label = consistency_match.group(1).lower()
        # Validate against the enum
        valid = {c.value for c in ConsistencyLabel}
        result["consistency"] = label if label in valid else ConsistencyLabel.UNCERTAIN.value

    # CONFIDENCE
    confidence_match = re.search(
        r"CONFIDENCE\s*:\s*(-?[0-9]*\.?[0-9]+)",
        block,
        re.IGNORECASE,
    )
    if confidence_match:
        try:
            raw_conf = float(confidence_match.group(1))
            # Clamp to [0, 1] — model occasionally outputs values > 1
            result["confidence"] = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, raw_conf))
        except ValueError:
            pass  # keep fallback

    # KEY_FACTORS
    factors_match = re.search(
        r"KEY_FACTORS\s*:\s*(.+?)(?=\n[A-Z_]+\s*:|$|---END)",
        block,
        re.DOTALL | re.IGNORECASE,
    )
    if factors_match:
        raw_factors = factors_match.group(1).strip()
        result["key_factors"] = [
            f.strip()
            for f in raw_factors.split(",")
            if f.strip()
        ]


def _extract_fields_loose(text: str, result: Dict[str, Any]) -> None:
    """Fallback field extraction when structured delimiters are missing.

    Searches the full model output for KEY: value patterns.
    Less precise than block extraction, but catches most format deviations.

    Modifies result dict in-place.

    Args:
        text:   Full model output string.
        result: Dict to update.
    """
    # Try the same regexes on the full text instead of the block
    _extract_fields_from_block(text, result)
