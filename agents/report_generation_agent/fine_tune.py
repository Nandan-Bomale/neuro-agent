"""
fine_tune.py  —  LoRA Fine-Tuning Script for Report Generation Agent
─────────────────────────────────────────────────────────────────────
Runs on the college DGX system (≥16GB VRAM).
Do NOT run on the RTX 3050 laptop — it will OOM.

Overview
────────
This script fine-tunes either:
  • microsoft/Phi-3-mini-4k-instruct
  • meta-llama/Llama-3.2-3B-Instruct

using QLoRA (4-bit base model + LoRA adapters) via the TRL SFTTrainer.

The training task: given a structured input block (patient data + vision
finding + RAG citations + clinical history), generate a structured 5-section
radiology report.

After training:
  1. LoRA adapter weights are saved to --output_dir/
  2. Optionally merged with the base model for a standalone model
  3. Download adapter weights to your laptop for inference

Workflow
────────
  # On DGX (after cloning the repo and installing requirements):
  python agents/report_generation_agent/fine_tune.py \\
      --model_name microsoft/Phi-3-mini-4k-instruct \\
      --dataset_path data/fine_tune/radiology_reports.jsonl \\
      --output_dir models/llm/phi3_report_lora \\
      --num_epochs 3 \\
      --batch_size 4 \\
      --hf_token YOUR_TOKEN

  # After training — merge adapter into base model (optional):
  python agents/report_generation_agent/fine_tune.py \\
      --merge_only \\
      --adapter_path models/llm/phi3_report_lora \\
      --output_dir models/llm/phi3_report_merged

  # On laptop — inference with the LoRA adapter (no merge needed):
  # Just set model_name to the base model and pass adapter_path to LLMLoader.

Dataset format (JSONL)
──────────────────────
Each line in the JSONL file is one training example:
  {
    "messages": [
      {"role": "system",    "content": "<system prompt>"},
      {"role": "user",      "content": "<structured input block>"},
      {"role": "assistant", "content": "<structured 5-section report>"}
    ]
  }

See DatasetBuilder.generate_synthetic_dataset() for how to create this.

Requirements (DGX only)
───────────────────────
  pip install trl>=0.8.6 peft>=0.10.0 bitsandbytes>=0.43.0
  pip install accelerate>=0.29.0 transformers>=4.40.0 datasets>=2.19.0
  pip install wandb  # optional but recommended for loss tracking
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LoRA target modules per model family
# (which weight matrices to apply LoRA to)
# ─────────────────────────────────────────────────────────────────────────────

LORA_TARGET_MODULES = {
    "phi3": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "llama": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Training configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FineTuneConfig:
    """All hyperparameters in one place — easy to log and reproduce."""

    # Model
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    hf_token: Optional[str] = None

    # Data
    dataset_path: str = "data/fine_tune/radiology_reports.jsonl"
    val_split: float = 0.1                  # fraction held out for validation
    max_seq_length: int = 2048              # max tokens per example

    # LoRA
    lora_r: int = 16                        # rank — higher = more params, better quality
    lora_alpha: int = 32                    # scaling factor (typically 2x rank)
    lora_dropout: float = 0.05
    lora_bias: str = "none"

    # Training
    output_dir: str = "models/llm/phi3_report_lora"
    num_epochs: int = 3
    batch_size: int = 4                     # per-device batch size on DGX
    gradient_accumulation_steps: int = 4   # effective batch = 4 * 4 = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    fp16: bool = False                      # DGX A100 supports bf16 — prefer that
    bf16: bool = True

    # Logging + saving
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3              # keep only last 3 checkpoints
    load_best_model_at_end: bool = True
    report_to: str = "wandb"              # set to "none" to disable wandb

    # Reproducibility
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Model + tokenizer loading for fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model_for_training(
    config: FineTuneConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the base model in 4-bit + prepare it for LoRA training.

    Steps:
      1. Load in 4-bit NF4 (same quant as inference, but with training support)
      2. Call prepare_model_for_kbit_training() — enables gradient checkpointing
         and casts non-quantized layers to float32 for stable training
      3. Apply LoRA adapters via get_peft_model()
    """
    model_family = "phi3" if "phi" in config.model_name.lower() else "llama"
    trust_remote = model_family == "phi3"

    logger.info("Loading tokenizer: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=config.hf_token,
        trust_remote_code=trust_remote,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # For training: pad on the right so the loss mask aligns correctly
    tokenizer.padding_side = "right"

    logger.info("Loading base model in 4-bit NF4 …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=config.hf_token,
        trust_remote_code=trust_remote,
        attn_implementation="eager",        # flash-attn2 can conflict with LoRA
    )

    # Required before applying LoRA to a quantized model:
    # - Enables gradient checkpointing (saves VRAM during backward pass)
    # - Casts LayerNorm + LM head to float32 so gradients don't vanish
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    logger.info("Applying LoRA adapters (r=%d, alpha=%d) …", config.lora_r, config.lora_alpha)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=LORA_TARGET_MODULES[model_family],
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_and_format_dataset(
    config: FineTuneConfig,
    tokenizer: AutoTokenizer,
) -> tuple[Dataset, Dataset]:
    """
    Load the JSONL dataset and format each example using the model's
    chat template.

    Each JSONL example has the format:
        {"messages": [{"role": "system", ...}, {"role": "user", ...},
                      {"role": "assistant", ...}]}

    Returns (train_dataset, eval_dataset).
    """
    logger.info("Loading dataset from: %s", config.dataset_path)

    dataset = load_dataset(
        "json",
        data_files=config.dataset_path,
        split="train",
    )
    logger.info("Loaded %d examples.", len(dataset))

    # Apply chat template to format messages → a single string per example
    def apply_template(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,   # False for training (labels include assistant)
        )
        return {"text": text}

    dataset = dataset.map(apply_template, num_proc=4)

    # Train / validation split
    split = dataset.train_test_split(
        test_size=config.val_split,
        seed=config.seed,
    )
    return split["train"], split["test"]


# ─────────────────────────────────────────────────────────────────────────────
# Trainer setup
# ─────────────────────────────────────────────────────────────────────────────

def build_trainer(
    config: FineTuneConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> SFTTrainer:
    """
    Configure and return a TRL SFTTrainer.

    Key choices:
    - DataCollatorForCompletionOnlyLM: masks the loss on the prompt tokens
      so we only train the model to predict the ASSISTANT turn (the report).
      This is critical — without it, the model wastes capacity learning to
      reproduce the system prompt and user input.
    - packing=False: each example is padded individually. Packing is faster
      but harder to control for medical text where example lengths vary a lot.
    """
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=config.report_to,
        run_name="neuroagent-report-lora",
        seed=config.seed,
        # Memory optimizations for DGX:
        dataloader_pin_memory=True,
        group_by_length=True,           # group similar-length examples → less padding waste
        optim="paged_adamw_32bit",      # paged optimizer frees VRAM during optimizer step
    )

    # Completion-only collator: compute loss only on the assistant response tokens.
    # The response template string must match what the chat template produces.
    response_template = _get_response_template(config.model_name)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        data_collator=data_collator,
        max_seq_length=config.max_seq_length,
        packing=False,
        args=training_args,
    )

    return trainer


def _get_response_template(model_name: str) -> str:
    """
    Return the token string that marks the start of the assistant's turn.
    The DataCollator uses this to find where the assistant response begins
    so it can mask the prompt from the loss.
    """
    lower = model_name.lower()
    if "phi" in lower:
        return "<|assistant|>"
    elif "llama" in lower:
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        return "### Response:"


# ─────────────────────────────────────────────────────────────────────────────
# Post-training: save and optionally merge
# ─────────────────────────────────────────────────────────────────────────────

def save_adapter(trainer: SFTTrainer, output_dir: str) -> None:
    """
    Save the LoRA adapter weights and tokenizer.

    What gets saved:
      adapter_config.json      — LoRA hyperparameters
      adapter_model.safetensors — the trained weight deltas (~50-150MB)
      tokenizer files          — needed for inference

    To use on the laptop:
      from peft import PeftModel
      model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, ...)
      model = PeftModel.from_pretrained(model, output_dir)
    """
    logger.info("Saving LoRA adapter to: %s", output_dir)
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
    logger.info("Adapter saved.")


def merge_adapter_into_base(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    hf_token: Optional[str] = None,
) -> None:
    """
    Merge LoRA adapter weights into the base model and save a standalone
    full-precision model.

    When to use:
      - If you want a single model file with no PEFT dependency
      - For sharing the model without requiring the original base weights

    When NOT to use (keep adapter separate):
      - Inference on laptop with 4-bit quantization (load base in 4-bit,
        load adapter on top — this is more VRAM-efficient than a merged model)
      - When you want to swap adapters without reloading the base model

    The merged model is saved in float16 to keep file size manageable.
    """
    from peft import PeftModel

    logger.info("Loading base model for merge: %s", base_model_name)
    trust_remote = "phi" in base_model_name.lower()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=trust_remote,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        token=hf_token,
        trust_remote_code=trust_remote,
    )

    logger.info("Loading adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging adapter into base model weights …")
    model = model.merge_and_unload()

    logger.info("Saving merged model to: %s", output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    logger.info("Merged model saved. Size: %.1f GB",
                sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset builder
# ─────────────────────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Build or validate a fine-tuning dataset for the Report Agent.

    For fine-tuning a radiology report writer, you need training examples
    in the form:
        (patient data + vision finding + RAG citations + clinical history)
            → structured 5-section radiology report

    Since real paired (MRI + report) datasets are hard to access, we use:
      1. Synthetic patient profiles paired with BraTS findings
      2. Template-based report generation with controlled variation
      3. (Later) real reports from MIMIC-CXR or similar datasets

    This class generates the JSONL training file.
    """

    # Canonical findings that the Vision Agent can produce
    FINDINGS = [
        {"label": "Glioma", "grade": "high-grade", "confidence_range": (0.78, 0.96)},
        {"label": "Glioma", "grade": "low-grade",  "confidence_range": (0.60, 0.80)},
        {"label": "Meningioma",                     "confidence_range": (0.72, 0.91)},
        {"label": "Metastatic lesion",              "confidence_range": (0.65, 0.88)},
        {"label": "No abnormality detected",        "confidence_range": (0.85, 0.98)},
    ]

    LOCATIONS = [
        "right frontal lobe",
        "left temporal lobe",
        "right parietal lobe",
        "left occipital lobe",
        "right cerebellar hemisphere",
        "left basal ganglia",
        "right thalamus",
        "parasagittal region",
    ]

    SYMPTOM_SETS = [
        ["progressive headache", "nausea", "vomiting"],
        ["focal seizure", "left arm weakness"],
        ["blurred vision", "diplopia"],
        ["memory impairment", "personality changes"],
        ["headache", "papilloedema"],
        ["right-sided weakness", "dysphasia"],
        ["ataxia", "dysmetria"],
        ["no symptoms — incidental finding on routine scan"],
    ]

    HISTORY_SETS = [
        ["hypertension", "type 2 diabetes"],
        ["no significant neurological history"],
        ["previous breast carcinoma (5 years, in remission)"],
        ["epilepsy (childhood onset)"],
        ["hypertension"],
        ["migraine (chronic)"],
        ["previous lung adenocarcinoma (3 years, post-resection)"],
    ]

    # Minimal citation templates (to be replaced with real PubMed data)
    CITATION_TEMPLATES = {
        "Glioma": [
            {
                "title": "CBTRUS Statistical Report: Primary Brain and Other CNS Tumors",
                "pmid": "34608945", "year": 2021,
                "key_finding": "Glioblastoma is the most common malignant primary brain tumour in adults, peak incidence in the sixth decade.",
            },
            {
                "title": "WHO Classification of Tumours of the Central Nervous System (5th ed.)",
                "pmid": "34255900", "year": 2021,
                "key_finding": "Updated WHO classification integrates molecular markers with histological features for glioma grading.",
            },
        ],
        "Meningioma": [
            {
                "title": "Epidemiology of intracranial meningiomas",
                "pmid": "22248535", "year": 2012,
                "key_finding": "Meningiomas are the most common primary intracranial tumour, with female predominance (2:1) and peak incidence in the 6th–7th decades.",
            },
        ],
        "Metastatic lesion": [
            {
                "title": "Brain metastases: epidemiology and pathophysiology",
                "pmid": "19520089", "year": 2009,
                "key_finding": "Brain metastases occur in 20–40% of cancer patients; lung, breast, and melanoma are the most common primary tumours.",
            },
        ],
        "No abnormality detected": [
            {
                "title": "MRI of the brain: normal variants and pitfalls",
                "pmid": "23034024", "year": 2012,
                "key_finding": "A normal brain MRI effectively excludes most structural causes of headache; clinical correlation remains essential.",
            },
        ],
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_synthetic_dataset(
        self,
        output_path: str,
        n_examples: int = 500,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Generate a JSONL fine-tuning dataset with synthetic radiology cases.

        Each example is a complete (system, user, assistant) conversation.
        The assistant turn is a templated but varied 5-section radiology report.

        Parameters
        ----------
        output_path : str — where to write the JSONL file
        n_examples  : int — number of training examples to generate
        system_prompt : str | None — if None, uses SYSTEM_PROMPT_REPORT
        """
        # Import lazily so this file can be imported without the report agent loaded
        from agents.report_generation_agent.prompt_templates import (
            SYSTEM_PROMPT_REPORT,
            PromptBuilder,
            PatientData,
            VisionOutput,
            RAGOutput,
            RAGCitation,
            ClinicalOutput,
        )

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT_REPORT

        builder = PromptBuilder()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Generating %d synthetic training examples → %s",
            n_examples, output_path
        )

        with open(output_path, "w", encoding="utf-8") as f:
            for i in range(n_examples):
                example = self._generate_one_example(builder, system_prompt)
                f.write(json.dumps(example) + "\n")

                if (i + 1) % 100 == 0:
                    logger.info("  Generated %d / %d examples …", i + 1, n_examples)

        logger.info("Dataset saved to %s", output_path)

    def _generate_one_example(
        self,
        builder,
        system_prompt: str,
    ) -> dict:
        """Generate one (system, user, assistant) training example."""
        from agents.report_generation_agent.prompt_templates import (
            PatientData, VisionOutput, RAGOutput, RAGCitation, ClinicalOutput,
        )

        # Sample random patient
        finding_template = random.choice(self.FINDINGS)
        label        = finding_template["label"]
        confidence   = round(random.uniform(*finding_template["confidence_range"]), 2)
        location     = random.choice(self.LOCATIONS)
        age          = random.randint(28, 78)
        sex          = random.choice(["Male", "Female"])
        symptoms     = random.choice(self.SYMPTOM_SETS)
        history      = random.choice(self.HISTORY_SETS)

        grade_note = ""
        if "grade" in finding_template:
            grade_note = f" ({finding_template['grade']})"

        patient = PatientData(
            patient_id=f"SYNTH-{random.randint(1000, 9999)}",
            age=age,
            sex=sex,
            presenting_symptoms=symptoms,
            medical_history=history,
            medications=random.choice([[], ["dexamethasone 4mg"], ["amlodipine 5mg"]]),
            scan_date="2026-01-01",
        )

        vision = VisionOutput(
            finding=label + grade_note,
            confidence=confidence,
            location=location,
            size_cm=round(random.uniform(1.2, 5.5), 1) if label != "No abnormality detected" else None,
            characteristics=(
                random.sample(
                    ["heterogeneous T2 signal", "ring enhancement", "perilesional oedema",
                     "mass effect", "homogeneous enhancement", "cystic component",
                     "calcification", "restricted diffusion"],
                    k=random.randint(2, 4),
                ) if label != "No abnormality detected" else []
            ),
            gradcam_available=True,
        )

        # Citations for this finding type
        raw_cites = self.CITATION_TEMPLATES.get(
            label.split(" (")[0],  # strip grade note
            self.CITATION_TEMPLATES["No abnormality detected"]
        )
        citations = [
            RAGCitation(
                title=c["title"],
                pmid=c["pmid"],
                year=c["year"],
                relevance_score=round(random.uniform(0.75, 0.97), 2),
                key_finding=c["key_finding"],
            )
            for c in raw_cites
        ]

        rag = RAGOutput(
            query_used=f"{label} brain MRI findings management",
            citations=citations,
        )

        consistent = confidence >= 0.65
        clinical = ClinicalOutput(
            clinical_reasoning=(
                f"The patient's presenting symptoms ({', '.join(symptoms[:2])}) "
                f"in a {age}-year-old {sex.lower()} are {'consistent' if consistent else 'partially consistent'} "
                f"with the imaging finding of {label} at {location}."
            ),
            risk_factors_identified=[
                f"Age {age}" + (" (>50: elevated risk for high-grade tumours)" if age > 50 else ""),
                f"Sex: {sex}",
                symptoms[0] if symptoms else "Non-specific headache",
            ],
            history_consistent_with_finding=consistent,
            notes="",
        )

        _, user_message = builder.build_report_prompt(
            vision_output=vision,
            rag_output=rag,
            clinical_output=clinical,
            patient=patient,
        )

        # Build the assistant response (structured report)
        assistant_response = self._build_report_text(
            label, grade_note, location, vision, clinical, citations, confidence
        )

        return {
            "messages": [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": user_message},
                {"role": "assistant", "content": assistant_response},
            ]
        }

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 0.80:
            return "HIGH"
        elif score >= 0.55:
            return "MODERATE"
        return "LOW"

    def _build_report_text(
        self,
        label: str,
        grade_note: str,
        location: str,
        vision,
        clinical,
        citations: list,
        confidence: float,
    ) -> str:
        """
        Build a template-based structured report for training.
        Includes deliberate variation in phrasing to prevent overfitting.
        """
        conf_label = self._confidence_label(confidence)
        requires_review = conf_label in ("LOW", "MODERATE")
        review_note = (
            "\n⚠ Flag for mandatory radiologist review." if requires_review else ""
        )

        # FINDINGS
        if label == "No abnormality detected":
            findings = (
                "No significant intracranial abnormality is identified on this study. "
                "The brain parenchyma demonstrates normal signal characteristics throughout. "
                "No mass lesion, haemorrhage, infarct, or midline shift is detected. "
                "Ventricles and sulci are appropriate for the patient's age."
            )
            impression = (
                f"Normal brain MRI study. No structural correlate identified for the "
                f"patient's presenting symptoms. Clinical correlation is recommended."
            )
            recommendation = (
                "No urgent neurological intervention indicated on the basis of current imaging. "
                "Clinical follow-up is recommended to address the presenting symptoms. "
                "Consider repeat MRI with gadolinium contrast if symptoms persist or progress."
            )
        else:
            size_str = f"{vision.size_cm} cm " if vision.size_cm else ""
            chars = "; ".join(vision.characteristics) if vision.characteristics else "heterogeneous signal"
            findings = (
                f"A {size_str}lesion is identified in the {location}, demonstrating {chars}. "
                f"The imaging characteristics are in keeping with a {label}{grade_note}. "
                f"{'Mass effect is present on the adjacent parenchyma. ' if vision.size_cm and vision.size_cm > 3 else ''}"
                f"The remainder of the imaged brain demonstrates no additional focal abnormality."
            )
            impression = (
                f"The imaging findings are most consistent with a {label}{grade_note} "
                f"arising in the {location}. "
                f"{'The clinical history is consistent with this impression. ' if clinical.history_consistent_with_finding else ''}"
                f"Histological confirmation is required for definitive diagnosis."
            )
            recommendation = (
                f"{'Urgent' if confidence > 0.80 else 'Prompt'} neurosurgical consultation is recommended. "
                f"MRI with gadolinium contrast is advised if not already performed. "
                f"Multidisciplinary team review is indicated.{review_note}"
            )

        # SUPPORTING EVIDENCE
        evidence_lines = []
        for i, c in enumerate(citations, start=1):
            evidence_lines.append(
                f"[{i}] ({c.year}). {c.title}. PMID: {c.pmid}. "
                f"Relevance: {c.relevance_score:.2f}.\n"
                f"    Key finding: {c.key_finding}"
            )
        evidence = "\n".join(evidence_lines) if evidence_lines else "No citations available."

        # CONFIDENCE
        rationale = (
            f"{'Strong' if confidence >= 0.80 else 'Moderate'} concordance between "
            f"imaging phenotype and clinical presentation supports a confidence "
            f"score of {confidence:.2f}."
        )

        return (
            f"## FINDINGS\n{findings}\n\n"
            f"## IMPRESSION\n{impression}\n\n"
            f"## SUPPORTING EVIDENCE\n{evidence}\n\n"
            f"## CONFIDENCE\n"
            f"Level: {conf_label}\n"
            f"Score: {confidence:.2f}\n"
            f"Rationale: {rationale}\n\n"
            f"## RECOMMENDATION\n{recommendation}"
        )

    def validate_dataset(self, dataset_path: str) -> dict:
        """
        Sanity-check a JSONL dataset before training.

        Checks:
          - All examples have "messages" key
          - Each message has "role" and "content"
          - All three roles (system, user, assistant) present
          - No empty content fields
          - Reports token count distribution

        Returns a summary dict.
        """
        issues = []
        n_examples = 0
        char_counts = []

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    example = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    issues.append(f"Line {line_num}: JSON parse error — {e}")
                    continue

                n_examples += 1

                if "messages" not in example:
                    issues.append(f"Line {line_num}: missing 'messages' key")
                    continue

                roles = {m["role"] for m in example["messages"]}
                for required_role in ("system", "user", "assistant"):
                    if required_role not in roles:
                        issues.append(f"Line {line_num}: missing role '{required_role}'")

                for msg in example["messages"]:
                    if not msg.get("content", "").strip():
                        issues.append(f"Line {line_num}: empty content for role '{msg['role']}'")

                total_chars = sum(len(m["content"]) for m in example["messages"])
                char_counts.append(total_chars)

        summary = {
            "total_examples": n_examples,
            "issues_found": len(issues),
            "issues": issues[:10],  # show first 10
            "avg_chars_per_example": int(sum(char_counts) / max(len(char_counts), 1)),
            "min_chars": min(char_counts) if char_counts else 0,
            "max_chars": max(char_counts) if char_counts else 0,
            "approx_avg_tokens": int(sum(char_counts) / max(len(char_counts), 1)) // 4,
        }
        return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: FineTuneConfig) -> None:
    """Full training run."""
    logger.info("=" * 60)
    logger.info("NeuroAgent — Report Agent LoRA Fine-Tuning")
    logger.info("Model:      %s", config.model_name)
    logger.info("Dataset:    %s", config.dataset_path)
    logger.info("Output dir: %s", config.output_dir)
    logger.info("Epochs:     %d", config.num_epochs)
    logger.info("Batch size: %d (×%d grad accum = %d effective)",
                config.batch_size, config.gradient_accumulation_steps,
                config.batch_size * config.gradient_accumulation_steps)
    logger.info("=" * 60)

    # 1. Load model
    model, tokenizer = load_base_model_for_training(config)

    # 2. Load and format dataset
    train_dataset, eval_dataset = load_and_format_dataset(config, tokenizer)
    logger.info(
        "Dataset split — train: %d, eval: %d",
        len(train_dataset), len(eval_dataset),
    )

    # 3. Build trainer
    trainer = build_trainer(
        config, model, tokenizer, train_dataset, eval_dataset
    )

    # 4. Train
    logger.info("Starting training …")
    trainer.train()

    # 5. Save
    save_adapter(trainer, config.output_dir)
    logger.info("Training complete. Adapter saved to: %s", config.output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroAgent — Fine-tune the Report Generation Agent LLM with QLoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── train ──────────────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Run LoRA fine-tuning on DGX")
    train_parser.add_argument("--model_name",      default="microsoft/Phi-3-mini-4k-instruct")
    train_parser.add_argument("--dataset_path",    default="data/fine_tune/radiology_reports.jsonl")
    train_parser.add_argument("--output_dir",      default="models/llm/phi3_report_lora")
    train_parser.add_argument("--num_epochs",      type=int,   default=3)
    train_parser.add_argument("--batch_size",      type=int,   default=4)
    train_parser.add_argument("--grad_accum",      type=int,   default=4)
    train_parser.add_argument("--learning_rate",   type=float, default=2e-4)
    train_parser.add_argument("--lora_r",          type=int,   default=16)
    train_parser.add_argument("--lora_alpha",      type=int,   default=32)
    train_parser.add_argument("--max_seq_length",  type=int,   default=2048)
    train_parser.add_argument("--hf_token",        default=None)
    train_parser.add_argument("--no_wandb",        action="store_true",
                              help="Disable Weights & Biases logging")
    train_parser.add_argument("--seed",            type=int,   default=42)

    # ── merge ──────────────────────────────────────────────────────────────────
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge LoRA adapter into base model (creates standalone model)"
    )
    merge_parser.add_argument("--model_name",    required=True)
    merge_parser.add_argument("--adapter_path",  required=True)
    merge_parser.add_argument("--output_dir",    required=True)
    merge_parser.add_argument("--hf_token",      default=None)

    # ── generate-data ──────────────────────────────────────────────────────────
    data_parser = subparsers.add_parser(
        "generate-data",
        help="Generate synthetic training dataset (JSONL)"
    )
    data_parser.add_argument("--output_path",  default="data/fine_tune/radiology_reports.jsonl")
    data_parser.add_argument("--n_examples",   type=int, default=500)
    data_parser.add_argument("--seed",         type=int, default=42)

    # ── validate-data ──────────────────────────────────────────────────────────
    val_parser = subparsers.add_parser(
        "validate-data",
        help="Validate a JSONL dataset file before training"
    )
    val_parser.add_argument("--dataset_path", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command is None:
        print("Usage: python fine_tune.py {train|merge|generate-data|validate-data} --help")
        sys.exit(1)

    if args.command == "train":
        config = FineTuneConfig(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            max_seq_length=args.max_seq_length,
            hf_token=args.hf_token or os.getenv("HF_TOKEN"),
            report_to="none" if args.no_wandb else "wandb",
            seed=args.seed,
        )
        train(config)

    elif args.command == "merge":
        merge_adapter_into_base(
            base_model_name=args.model_name,
            adapter_path=args.adapter_path,
            output_dir=args.output_dir,
            hf_token=args.hf_token or os.getenv("HF_TOKEN"),
        )

    elif args.command == "generate-data":
        db = DatasetBuilder(seed=args.seed)
        db.generate_synthetic_dataset(
            output_path=args.output_path,
            n_examples=args.n_examples,
        )

    elif args.command == "validate-data":
        db = DatasetBuilder()
        summary = db.validate_dataset(args.dataset_path)
        print(json.dumps(summary, indent=2))
        if summary["issues_found"] > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
