"""
Fine-tune YOLO11n on brain-tumor dataset.
Windows-safe: wrapped in if __name__ == '__main__'.
"""
from pathlib import Path
import shutil


def main():
    from ultralytics import YOLO

    print("Fine-tuning YOLO11n for brain tumor detection (detect task)...")

    # Standard detect model — brain-tumor.yaml only has boxes, not segments
    model = YOLO("yolo11n.pt")
    print(f"Model task: {model.task}")

    results = model.train(
        data="brain-tumor.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,
        patience=15,
        save=True,
        save_period=20,
        project="models/yolo_finetune",
        name="brain_tumor_detect_v1",
        exist_ok=True,
        verbose=True,
        amp=True,
        lr0=0.01,
        warmup_epochs=3,
        cos_lr=True,
        degrees=15.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        close_mosaic=10,
        workers=0,      # Windows: disable multiprocessing workers
    )

    print("\n=== Training Complete ===")
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        dest = Path("models/yolo/weights/detect_best.pt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, dest)
        print(f"Saved to: {dest}")
        metrics = results.results_dict
        map50 = metrics.get("metrics/mAP50(B)", "N/A")
        map95 = metrics.get("metrics/mAP50-95(B)", "N/A")
        if isinstance(map50, float):
            print(f"mAP50:    {map50:.4f}")
        if isinstance(map95, float):
            print(f"mAP50-95: {map95:.4f}")
    else:
        print("WARNING: best.pt not found")


if __name__ == "__main__":
    main()
