import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_models():
    print("==================================================")
    print("      Downloading AI Models from Hugging Face     ")
    print("==================================================")
    
    repo_id = "cloudxvi/neuro-agent-models"
    models_to_check = [
        "models/tumor_classifier/type_ensemble_best.pth",
        "models/tumor_classifier/type_ensemble_classes.json",
        "models/tumor_classifier/grade_classifier_best.pth",
        "models/tumor_classifier/grade_classifier_classes.json",
        "models/yolo/weights/detect_best.pt",
        "models/yolo/weights/best.pt",
        "models/bbox_regressor/bbox_model_best.pth",
        "models/bbox_regressor/labels.json",
        "models/radiogenomics/radiogenomics_best.pth"
    ]
    
    for i, rel in enumerate(models_to_check, 1):
        target = Path(rel)
        if not target.exists():
            print(f"[{i}/{len(models_to_check)}] Downloading {target.name}...")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(repo_id=repo_id, filename=rel, local_dir=".")
                print(f"   ? Successfully downloaded {target.name}")
            except Exception as exc:
                print(f"   ? Error downloading {target.name}: {exc}")
        else:
            print(f"[{i}/{len(models_to_check)}] ? {target.name} already exists.")
            
    print("\nAll models are ready!")

if __name__ == "__main__":
    download_models()
