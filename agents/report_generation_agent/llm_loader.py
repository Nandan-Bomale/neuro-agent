"""
llm_loader.py
─────────────
Shared 4-bit quantized LLM loader for NeuroAgent.

Supports:
  • microsoft/Phi-3-mini-4k-instruct   (laptop — ~2.5 GB VRAM in 4-bit)
  • meta-llama/Llama-3.2-3B-Instruct   (DGX fine-tuned weights — ~3 GB VRAM in 4-bit)

Usage
-----
    from agents.report_generation_agent.llm_loader import LLMLoader

    loader = LLMLoader(model_name="microsoft/Phi-3-mini-4k-instruct")
    model, tokenizer = loader.load()

    # Generate text
    response = loader.generate(
        system_prompt="You are a radiology assistant.",
        user_message="Describe the findings.",
        max_new_tokens=512,
    )

Design decisions
----------------
  - Singleton pattern: model is loaded once and cached; subsequent calls to
    LLMLoader() with the same model_name return the cached instance.
  - 4-bit NF4 quantization via BitsAndBytesConfig (bitsandbytes ≥ 0.43).
  - device_map="auto" so the same script works on your RTX 3050 (single GPU)
    and on the DGX (multi-GPU) without any code changes.
  - Flash Attention 2 is enabled when available; silently falls back to
    standard attention so the code runs everywhere.
  - Chat-template formatting is handled internally; callers just pass
    system_prompt + user_message strings.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

logger = logging.getLogger(__name__)

# ── Supported models ──────────────────────────────────────────────────────────

SUPPORTED_MODELS: dict[str, dict] = {
    "microsoft/Phi-3-mini-4k-instruct": {
        "context_length": 4096,
        "chat_template": "phi3",
        "trust_remote_code": True,
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "context_length": 8192,
        "chat_template": "llama3",
        "trust_remote_code": False,
    },
}

# Default model for laptop inference
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# ── Singleton registry ────────────────────────────────────────────────────────

_loader_registry: dict[str, "LLMLoader"] = {}


# ── BitsAndBytes 4-bit config ─────────────────────────────────────────────────

def _build_bnb_config() -> BitsAndBytesConfig:
    """
    NF4 quantization config.
    - load_in_4bit: quantize weights to 4-bit on load
    - bnb_4bit_quant_type: NF4 (NormalFloat4) — best quality for LLMs
    - bnb_4bit_use_double_quant: nested quantization saves ~0.4 GB extra
    - bnb_4bit_compute_dtype: bfloat16 for computation (fp16 on older GPUs)
    """
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


# ── Main loader class ─────────────────────────────────────────────────────────

class LLMLoader:
    """
    Singleton 4-bit LLM loader.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or path to local fine-tuned weights.
        Must be one of SUPPORTED_MODELS or a local path whose base
        name matches a supported model family ("phi" or "llama").
    hf_token : str | None
        HuggingFace access token for gated models (e.g. Llama).
        Falls back to the HF_TOKEN environment variable if not provided.
    offload_to_cpu : bool
        If True, layers that don't fit in VRAM are offloaded to CPU RAM.
        Useful on the laptop when VRAM is tight. Slower but never OOMs.
    """

    def __new__(
        cls,
        model_name: str = DEFAULT_MODEL,
        hf_token: Optional[str] = None,
        offload_to_cpu: bool = False,
    ) -> "LLMLoader":
        # Return cached instance if already loaded
        if model_name in _loader_registry:
            logger.info(
                "LLMLoader: returning cached instance for '%s'", model_name
            )
            return _loader_registry[model_name]

        instance = super().__new__(cls)
        instance._initialized = False
        _loader_registry[model_name] = instance
        return instance

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        hf_token: Optional[str] = None,
        offload_to_cpu: bool = False,
    ) -> None:
        if self._initialized:
            return  # Already set up by a previous __init__ call

        self.model_name = model_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.offload_to_cpu = offload_to_cpu

        # Resolve chat template type from model name
        self._chat_template = self._resolve_chat_template(model_name)
        self._context_length = self._resolve_context_length(model_name)

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._initialized = True

        logger.info(
            "LLMLoader initialised for '%s' (chat_template=%s, context=%d)",
            model_name,
            self._chat_template,
            self._context_length,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the model and tokenizer into VRAM (4-bit quantized).

        Returns
        -------
        (model, tokenizer) — both ready for inference.
        """
        if self.model is not None and self.tokenizer is not None:
            logger.info("LLMLoader: model already loaded, reusing.")
            return self.model, self.tokenizer

        logger.info("Loading tokenizer for '%s' …", self.model_name)
        self.tokenizer = self._load_tokenizer()

        logger.info("Loading model '%s' in 4-bit …", self.model_name)
        self.model = self._load_model()

        logger.info("Model loaded. VRAM usage: %s", self._vram_usage())
        return self.model, self.tokenizer

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Run a single chat-style generation.

        Parameters
        ----------
        system_prompt : str
            The system role prompt (sets model behaviour/persona).
        user_message : str
            The user turn — the actual clinical input.
        max_new_tokens : int
            Maximum tokens to generate. Default 512 — enough for a full
            radiology report. Increase to 1024 for verbose outputs.
        temperature : float
            Low temperature (0.1–0.3) for deterministic clinical text.
        do_sample : bool
            Set False for greedy decoding (fully deterministic).
        repetition_penalty : float
            Penalises repetitive phrases, common in fine-tuned medical LLMs.

        Returns
        -------
        str — the raw generated text (assistant turn only, stripped).
        """
        if self.model is None or self.tokenizer is None:
            self.load()

        # Build chat-formatted input
        input_ids = self._format_and_tokenize(system_prompt, user_message)

        # Check we're within context window
        n_input_tokens = input_ids.shape[-1]
        max_allowed = self._context_length - max_new_tokens
        if n_input_tokens > max_allowed:
            logger.warning(
                "Input has %d tokens, context limit is %d. "
                "Truncation may occur.",
                n_input_tokens,
                max_allowed,
            )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                generation_config=GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                ),
            )

        # Decode only the newly generated tokens (strip the prompt)
        new_tokens = output_ids[0, n_input_tokens:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def unload(self) -> None:
        """
        Free VRAM by deleting the model from GPU memory.
        Call this when switching between agents to reclaim memory.
        """
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("LLMLoader: model '%s' unloaded from VRAM.", self.model_name)

        if self.model_name in _loader_registry:
            del _loader_registry[self.model_name]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_tokenizer(self) -> AutoTokenizer:
        trust = SUPPORTED_MODELS.get(self.model_name, {}).get(
            "trust_remote_code", False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=trust,
        )
        # Ensure a pad token exists (Phi-3 and Llama both have eos but not pad)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        trust = SUPPORTED_MODELS.get(self.model_name, {}).get(
            "trust_remote_code", False
        )
        bnb_config = _build_bnb_config()

        # Try Flash Attention 2 first; fall back gracefully
        attn_impl = self._best_attention_impl()

        device_map = "auto"
        if self.offload_to_cpu:
            # Offload to CPU RAM when VRAM is very tight
            device_map = {
                "": "cpu",
                "model.embed_tokens": 0,
                "model.norm": 0,
                "lm_head": 0,
            }
            logger.info(
                "CPU offload enabled — inference will be slower but stable."
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            token=self.hf_token,
            trust_remote_code=trust,
            attn_implementation=attn_impl,
        )
        model.eval()  # Disable dropout for inference
        return model

    def _format_and_tokenize(
        self, system_prompt: str, user_message: str
    ) -> torch.Tensor:
        """
        Build the chat-formatted prompt and tokenize it.
        Handles Phi-3 and Llama-3 chat templates automatically.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Use the tokenizer's built-in chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and \
                self.tokenizer.chat_template is not None:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Manual fallback templates
            prompt = self._manual_chat_template(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._context_length,
        ).to(self.model.device)

        return inputs["input_ids"]

    def _manual_chat_template(self, messages: list[dict]) -> str:
        """
        Fallback chat templates for Phi-3 and Llama-3 if the tokenizer
        does not ship with a chat_template JSON.
        """
        if self._chat_template == "phi3":
            # Phi-3 format: <|system|>...<|end|><|user|>...<|end|><|assistant|>
            parts = []
            for msg in messages:
                role_tag = f"<|{msg['role']}|>"
                parts.append(f"{role_tag}\n{msg['content']}<|end|>")
            parts.append("<|assistant|>")
            return "\n".join(parts)

        elif self._chat_template == "llama3":
            # Llama-3 format: <|begin_of_text|><|start_header_id|>...<|eot_id|>
            bos = "<|begin_of_text|>"
            parts = [bos]
            for msg in messages:
                parts.append(
                    f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
                    f"{msg['content']}<|eot_id|>"
                )
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(parts)

        else:
            # Generic fallback
            system = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            user = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            )
            return f"System: {system}\n\nUser: {user}\n\nAssistant:"

    def _best_attention_impl(self) -> str:
        """
        Return the best available attention implementation.
        Flash Attention 2 requires Ampere+ GPUs (RTX 30xx and above — yours
        qualifies). Falls back to 'eager' if not available.
        """
        try:
            import flash_attn  # noqa: F401
            logger.info("Flash Attention 2 available — using it.")
            return "flash_attention_2"
        except ImportError:
            logger.info(
                "flash_attn not installed — using standard attention. "
                "Install with: pip install flash-attn --no-build-isolation"
            )
            return "eager"

    def _resolve_chat_template(self, model_name: str) -> str:
        if model_name in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[model_name]["chat_template"]
        # Infer from model name for local fine-tuned paths
        lower = model_name.lower()
        if "phi" in lower:
            return "phi3"
        if "llama" in lower:
            return "llama3"
        return "generic"

    def _resolve_context_length(self, model_name: str) -> int:
        if model_name in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[model_name]["context_length"]
        # Safe default
        return 4096

    def _vram_usage(self) -> str:
        if not torch.cuda.is_available():
            return "N/A (no CUDA)"
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        return f"{allocated:.2f} GB allocated / {reserved:.2f} GB reserved"

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @property
    def context_length(self) -> int:
        return self._context_length

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"LLMLoader(model='{self.model_name}', "
            f"template={self._chat_template}, "
            f"status={status})"
        )
