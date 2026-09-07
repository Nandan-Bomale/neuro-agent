"""
swap_yolo_weights.py — Run after fine-tuning completes.
Finds the best trained weights, tests on sample images, and
updates the vision_2d_agent to use the new model.
"""
from __future__ import annotations
from pathlib import Path
import shutil, sys, cv2, numpy as np

FINETUNE_DIR = Path("models/yolo_finetune/brain_tumor_detect_v1")
DEST = Path("models/yolo/weights/detect_best.pt")

# Also check the runs dir (ultralytics saves there too)
ALT_DIRS = [
    Path("runs/detect/models/yolo_finetune/brain_tumor_detect_v1"),
    Path("runs/detect/brain_tumor_detect_v1"),
]

def find_best_pt():
    candidates = [FINETUNE_DIR / "weights" / "best.pt"]
    for d in ALT_DIRS:
        candidates.append(d / "weights" / "best.pt")
    for c in candidates:
        if c.exists():
            return c
    return None


def test_new_model(weights: Path):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    print(f"\nLoaded model: {weights}")
    print(f"Classes: {model.names}")

    test_imgs = [
        "data/interim/preprocessing/preprocessed_1787547244.jpg",
        "data/interim/preprocessing/preprocessed_1787547218.jpg",
        "data/interim/preprocessing/preprocessed_1787547168.jpg",
        "data/interim/preprocessing/preprocessed_1787547007.jpg",
        "data/interim/preprocessing/preprocessed_1787544669.jpg",
    ]

    print("\n--- Detection Results ---")
    all_detected = 0
    for p in test_imgs:
        if not Path(p).exists():
            continue
        img = cv2.imread(p)
        results = model(img, conf=0.10, verbose=False)
        r = results[0]
        if r.boxes and len(r.boxes) > 0:
            best_conf = float(r.boxes.conf.max())
            all_detected += 1
            print(f"  ✅ {Path(p).name[-20:]} — detected (conf={best_conf:.3f})")
        else:
            print(f"  ❌ {Path(p).name[-20:]} — no detection")

    return all_detected


def update_agent(weights: Path):
    """Update the vision agent to use the new detect model as primary."""
    agent_path = Path("agents/vision_2d_agent/agent.py")
    content = agent_path.read_text(encoding="utf-8")

    old = '_YOLO_WEIGHTS = Path("models/yolo/weights/best.pt")'
    new = f'_YOLO_WEIGHTS = Path("{str(weights).replace(chr(92), "/")}")'

    if old in content:
        content = content.replace(old, new)
        agent_path.write_text(content, encoding="utf-8")
        print(f"\nAgent updated to use: {weights}")
    else:
        print(f"\nWARNING: Could not find weight path in agent.py to update")


if __name__ == "__main__":
    best = find_best_pt()
    if not best:
        print("ERROR: No fine-tuned weights found yet. Training may still be in progress.")
        sys.exit(1)

    # Copy to canonical location
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, DEST)
    print(f"Copied {best} -> {DEST}")

    # Test
    detected = test_new_model(DEST)
    print(f"\nDetected in {detected}/5 test images")

    if detected >= 4:
        update_agent(DEST)
        print("✅ Agent upgraded to fine-tuned model!")
    else:
        print("⚠️  Only detected in some images — keeping original model")
        print("   Fine-tuned weights saved, but agent NOT updated")
