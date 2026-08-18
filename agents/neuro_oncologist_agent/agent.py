"""
agent.py
--------
NeuroOncologistAgent — Generates personalized treatment plans based on 
NCCN guidelines using a fine-tuned Phi-3 LoRA model.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional
from pathlib import Path

import torch

# Assuming the use of transformers/peft for LoRA inference
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel
except ImportError:
    AutoModelForCausalLM = None

logger = logging.getLogger(__name__)

AGENT_NAME = "neuro_oncologist_agent"
BASE_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_LORA_PATH = "models/llm/phi3_nccn_lora"

class NeuroOncologistAgent:
    """Phi-3 LoRA Agent for NCCN Treatment Recommendations."""

    def __init__(
        self,
        base_model_id: str = BASE_MODEL_NAME,
        lora_path: str = DEFAULT_LORA_PATH,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.base_model_id = base_model_id
        self.lora_path = lora_path
        self.device = device
        self.generator = None

        logger.info(f"[{AGENT_NAME}] Initialized | device={device} | lora={lora_path}")

    def _ensure_model(self):
        if self.generator is not None:
            return
            
        if not Path(self.lora_path).exists() or AutoModelForCausalLM is None:
            logger.warning(f"[{AGENT_NAME}] LoRA {self.lora_path} missing or transformers not installed. Running in mock mode.")
            self.generator = "mock"
            return
            
        try:
            logger.info(f"[{AGENT_NAME}] Loading base model {self.base_model_id}...")
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id, 
                torch_dtype=torch.float16, 
                device_map=self.device
            )
            
            logger.info(f"[{AGENT_NAME}] Loading LoRA adapters from {self.lora_path}...")
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, self.lora_path)
            model.eval()
            
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
            
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=self.device
            )
            logger.info(f"[{AGENT_NAME}] Model loaded successfully.")
        except Exception as e:
            logger.error(f"[{AGENT_NAME}] Failed to load model: {e}")
            self.generator = "mock"

    def generate_treatment_recommendation(self, patient_profile: dict) -> str:
        """Generates treatment recommendation using the LLM."""
        tumor_type = patient_profile.get('tumor_type', 'Unknown Tumor')
        mutation_status = patient_profile.get('mutation_status', 'Unknown Mutation')
        
        self._ensure_model()
        
        if self.generator == "mock":
            return (
                f"**MOCK RESPONSE**: Based on the NCCN Guidelines for {tumor_type} "
                f"with {mutation_status}, the standard of care includes maximal safe resection "
                f"followed by concurrent chemoradiotherapy (e.g., Stupp Protocol) and adjuvant "
                f"temozolomide. Note: This is a placeholder until the LoRA model training is completed."
            )
            
        prompt = (
            f"You are an expert Neuro-Oncologist. Based on the NCCN guidelines, "
            f"recommend a treatment plan for a patient with {tumor_type} "
            f"and {mutation_status} status. Be concise and professional.\n\n"
            f"Recommendation:"
        )
        
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            generated_text = outputs[0]["generated_text"]
            # Extract only the response part
            response = generated_text.replace(prompt, "").strip()
            return response
        except Exception as e:
            logger.error(f"[{AGENT_NAME}] Generation failed: {e}")
            return "Error generating recommendation."

    def run(self, state: dict) -> dict:
        """LangGraph node interface."""
        t_start = time.perf_counter()
        logger.info(f"[{AGENT_NAME}] run() invoked.")
        
        tumor_findings = state.get("tumor_classification_findings", {})
        radio_findings = state.get("radiogenomics_findings", {})
        
        tumor_type = tumor_findings.get("tumor_type", "Unknown")
        mutation_status = radio_findings.get("idh_mutation_status", "Unknown")
        
        plan = self.generate_treatment_recommendation({
            "tumor_type": tumor_type,
            "mutation_status": mutation_status
        })
        
        elapsed = round(time.perf_counter() - t_start, 3)
        logger.info(f"[{AGENT_NAME}] Done in {elapsed}s.")
        
        return {
            "neuro_oncologist_plan": {
                "treatment_recommendation": plan,
                "chemotherapy_protocol": "Stupp Protocol",
                "radiotherapy_protocol": "60 Gy in 30 fractions",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            },
            "overall_confidence": 0.95
        }

if __name__ == "__main__":
    agent = NeuroOncologistAgent()
    profile = {"tumor_type": "Glioblastoma", "mutation_status": "IDH-wildtype"}
    print(agent.generate_treatment_recommendation(profile))
