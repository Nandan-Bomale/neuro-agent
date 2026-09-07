"""
Download Kaggle brain tumor dataset with YOLO bounding box labels.
Using the "Brain Tumor Detection" dataset which has 3929 images with 4 classes.
"""
import os, zipfile, urllib.request
from pathlib import Path

# Dataset: Brain Tumor MRI Dataset with bounding boxes in YOLO format
# Available at multiple sources - we use a direct approach via ultralytics datasets
print("Checking for Kaggle dataset...")

# Check if we already have data
data_dir = Path("data/yolo_train")
if data_dir.exists() and len(list(data_dir.rglob("*.jpg"))) > 100:
    print(f"Dataset already exists: {len(list(data_dir.rglob('*.jpg')))} images found")
else:
    # Try downloading via ultralytics hub dataset
    from ultralytics import settings
    print("Will use roboflow or manual download")
    print("Checking pip for roboflow...")
    import subprocess
    result = subprocess.run(["pip", "show", "roboflow"], capture_output=True, text=True)
    print(result.stdout or "roboflow not installed")
    
print("Done")
