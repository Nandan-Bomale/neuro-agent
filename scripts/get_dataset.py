"""Download brain tumor YOLO dataset from Roboflow (public, no API key needed for public datasets)."""
import os
from pathlib import Path

# Use the public Roboflow Universe dataset
# "Brain Tumor Detection" - 3929 images, 4 classes, YOLO format
# Workspace: roboflow-100 or similar public workspace

try:
    from roboflow import Roboflow
    # Public dataset - no API key needed for public access
    rf = Roboflow(api_key="YOUR_API_KEY")
    print("Roboflow needs API key for download")
except Exception as e:
    print(f"Roboflow auth needed: {e}")

# Alternative: Use ultralytics dataset download
print("Trying ultralytics dataset approach...")
from ultralytics import YOLO
import yaml

# Check if any existing data
data_paths = list(Path("data").rglob("*.yaml"))
print(f"Existing yaml files: {data_paths}")

# Download brain tumor detection data via ultralytics
# The dataset "brain-tumor" is in their hub
import subprocess
result = subprocess.run(
    ["python", "-c", "from ultralytics.utils import DATASETS_DIR; print(DATASETS_DIR)"],
    capture_output=True, text=True
)
print("Datasets dir:", result.stdout.strip())
