import os

filepath = 'agents/tumor_classification_agent/agent.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the fake override
override = """        # --- DEMO OVERRIDE ---
        if pred["tumor_type"] == "notumor" and vf and vf.get("tumour_detected", False):
            pred["tumor_type"] = "glioma"
            pred["confidence"] = 0.95
            pred["type_probabilities"]["glioma"] = 0.95
            pred["type_probabilities"]["notumor"] = 0.05
        # ---------------------

"""
content = content.replace(override, "")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Reverted fake patch.")
