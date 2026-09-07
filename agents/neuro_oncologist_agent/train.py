import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Define dummy dataset (for placeholder training purposes)
# In reality, this would be replaced with actual tumor board NCCN guideline JSON -> Text mapping
dummy_data = [
    {
        "text": "You are an expert Neuro-Oncologist. Based on the NCCN guidelines, recommend a treatment plan for a patient with Glioblastoma and IDH-wildtype status. Be concise and professional.\n\nRecommendation: Based on the NCCN Guidelines for Glioblastoma (IDH-wildtype), the standard of care includes maximal safe resection followed by concurrent chemoradiotherapy (Stupp Protocol) and adjuvant temozolomide."
    },
    {
        "text": "You are an expert Neuro-Oncologist. Based on the NCCN guidelines, recommend a treatment plan for a patient with Glioma and IDH-mutant status. Be concise and professional.\n\nRecommendation: Based on the NCCN Guidelines for lower-grade Glioma (IDH-mutant), standard of care often involves maximal safe resection. Depending on high-risk features, adjuvant radiation and chemotherapy (PCV or temozolomide) may be indicated."
    }
] * 50

def train(model_id: str, output_dir: str):
    print(f"Loading base model {model_id} for LoRA tuning...")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    dataset = Dataset.from_list(dummy_data)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    print("Beginning LoRA fine-tuning...")
    try:
        from transformers import Trainer, DataCollatorForLanguageModeling
    except ImportError:
        print("Please install trl and transformers")
        return
        
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="no", # Only save at the end for demo
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    trainer.train()
    
    print(f"Saving LoRA adapters to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print("Training complete! The NeuroOncologistAgent will now use this real model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output-dir", type=str, default="models/llm/phi3_nccn_lora")
    args = parser.parse_args()
    
    train(args.base_model, args.output_dir)
