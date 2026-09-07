"""
Fine-tune YOLO11 on ultralytics built-in brain-tumor dataset.
This dataset is hosted by Ultralytics and downloads automatically.
"""
from ultralytics import YOLO
from pathlib import Path
import yaml

print("Starting fine-tune of YOLO11 on brain-tumor dataset...")
print("Ultralytics will auto-download the brain-tumor.yaml dataset on first run.")

# Load the pre-trained model (our downloaded tumor detector as starting point)
model = YOLO("models/yolo/weights/best.pt")

print(f"Base model: {type(model.model).__name__}")
print(f"Classes: {model.names}")
print(f"Task: {model.task}")

# Fine-tune using Ultralytics brain-tumor dataset
# This downloads ~3900 images with tumor bounding boxes automatically
results = model.train(
    data="brain-tumor.yaml",       # Ultralytics built-in dataset
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,                       # GPU
    patience=10,                    # Early stopping
    save=True,
    save_period=10,
    project="models/yolo_finetune",
    name="brain_tumor_v1",
    exist_ok=True,
    verbose=True,
    amp=True,                       # Mixed precision for speed
    lr0=0.001,                      # Lower LR for fine-tuning
    warmup_epochs=3,
    cos_lr=True,
    augment=True,
    degrees=10.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=0.8,
    mixup=0.1,
    copy_paste=0.1,
    # Freeze first 10 layers (keep backbone, fine-tune heads)
    freeze=10,
)

print("\n=== Fine-tuning Complete ===")
print(f"Best weights: {results.save_dir}/weights/best.pt")
print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")

# Copy best weights to models/yolo
import shutil
best_pt = Path(results.save_dir) / "weights" / "best.pt"
if best_pt.exists():
    dest = Path("models/yolo/weights/finetuned_best.pt")
    shutil.copy(best_pt, dest)
    print(f"Saved fine-tuned weights to: {dest}")
