import argparse
import json
import os
import re

def parse_guidelines_to_dataset(input_file: str, output_file: str, system_prompt: str):
    """
    Reads a clinical guideline text file and converts it into a .jsonl dataset
    formatted for LoRA instruction tuning (e.g., for Phi-3).
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    # Read the text file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Heuristic splitting: split by double newlines to get paragraphs/sections
    # A more sophisticated script might use regex to find section headers
    sections = re.split(r'\n\s*\n', text)
    
    dataset = []
    
    for i, section in enumerate(sections):
        section = section.strip()
        if not section or len(section) < 50:
            # Skip empty or very short sections
            continue
            
        # Example formatting: We turn each section into an instruction/response pair.
        # For a real dataset, you might want an LLM to generate specific questions for this text,
        # but as a baseline we ask the model to explain or summarize the guideline section.
        
        # A simple instruction prompt for the model
        user_message = f"Please explain the guideline recommendations regarding this topic:\n\n{section[:100]}..."
        
        # We format it in the standard chatml / messages format widely used by HF TRL and Phi-3
        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"What are the NCCN guidelines for the following clinical scenario or topic?\n\nContext: {section[:200]}..."},
                {"role": "assistant", "content": section}
            ]
        }
        dataset.append(record)

    # Write to jsonl
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for record in dataset:
            out_f.write(json.dumps(record) + '\n')
            
    print(f"Successfully generated {len(dataset)} instruction pairs and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Format clinical guidelines text into a .jsonl dataset for LoRA.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file containing guidelines")
    parser.add_argument("--output", type=str, default="data/lora_dataset.jsonl", help="Path to the output .jsonl file")
    parser.add_argument("--system", type=str, 
                        default="You are an expert Neuro-Oncologist AI. You provide accurate, evidence-based recommendations based on NCCN Guidelines.",
                        help="System prompt to include in each record.")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    parse_guidelines_to_dataset(args.input, args.output, args.system)

if __name__ == "__main__":
    main()
