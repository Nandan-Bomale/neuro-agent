"""
verify_training.py
------------------
Verifies that tumor classification training completed successfully
and that checkpoints are valid and loadable.

Usage (run from repo root):
    python scripts/verify_training.py --stage type
    python scripts/verify_training.py --stage grade
    python scripts/verify_training.py --stage both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the repo root (parent of scripts/) is on the path so that
# `agents.*` imports work when running `python scripts/verify_training.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

SAVE_DIR = Path("models/tumor_classifier")

TYPE_CKPT  = SAVE_DIR / "type_ensemble_best.pth"
GRADE_CKPT = SAVE_DIR / "grade_classifier_best.pth"
TYPE_META  = SAVE_DIR / "type_ensemble_classes.json"
GRADE_META = SAVE_DIR / "grade_classifier_classes.json"

TARGET_TYPE_ACC  = 0.95   # flag if below this (target 98–99%)
TARGET_GRADE_ACC = 0.85   # flag if below this (target 90%+)


def _load_and_check(ckpt_path: Path, meta_path: Path, target_acc: float, label: str) -> bool:
    print(f"\n{'-'*55}")
    print(f"  Checking: {label}")
    print(f"{'-'*55}")

    ok = True

    # 1. Files exist
    for p in [ckpt_path, meta_path]:
        if p.exists():
            size_mb = p.stat().st_size / 1024**2
            print(f"  [OK] File exists: {p.name}  ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {p}")
            ok = False

    if not ok:
        return False

    # 2. Checkpoint loads cleanly
    try:
        device = torch.device("cpu")
        ckpt   = torch.load(str(ckpt_path), map_location=device)
        print(f"  [OK] Checkpoint loads without error")
    except Exception as e:
        print(f"  [ERR] Failed to load checkpoint: {e}")
        return False

    # 3. Required keys present
    for key in ["model_state", "epoch", "val_accuracy", "val_loss", "class_names"]:
        if key in ckpt:
            print(f"  [OK] Key '{key}' present: {ckpt[key] if key != 'model_state' else '<state_dict>'}")
        else:
            print(f"  [MISSING KEY] '{key}' not in checkpoint")
            ok = False

    # 4. Accuracy check
    val_acc = ckpt.get("val_accuracy", 0.0)
    if val_acc >= target_acc:
        print(f"  [OK] val_accuracy = {val_acc:.4f}  (target >= {target_acc})")
    else:
        print(f"  [WARN] val_accuracy = {val_acc:.4f} is BELOW target {target_acc}"
              f" — consider more epochs or lower LR")

    # 5. Class names match metadata JSON
    meta        = json.loads(meta_path.read_text())
    ckpt_names  = ckpt.get("class_names", [])
    meta_names  = meta.get("class_names", [])
    if ckpt_names == meta_names:
        print(f"  [OK] Class names match: {ckpt_names}")
    else:
        print(f"  [WARN] Mismatch — ckpt: {ckpt_names}  meta: {meta_names}")

    # 6. State dict can be loaded into the correct model
    try:
        if "type" in label.lower():
            from agents.tumor_classification_agent.model import build_type_ensemble
            model = build_type_ensemble(
                num_classes=len(ckpt_names), device=device
            )
        else:
            from agents.tumor_classification_agent.model import build_grade_classifier
            model = build_grade_classifier(
                num_classes=len(ckpt_names), device=device
            )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Quick forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        print(f"  [OK] Forward pass OK — output shape: {out.shape}")
    except Exception as e:
        print(f"  [ERR] State dict load / forward failed: {e}")
        ok = False

    status = "PASSED" if ok else "FAILED"
    print(f"\n  Result: {status}\n")
    return ok


def main():
    p = argparse.ArgumentParser(description="Verify tumor classifier training outputs")
    p.add_argument("--stage", choices=["type", "grade", "both"], default="both")
    args = p.parse_args()

    results = {}
    if args.stage in ("type", "both"):
        results["type"] = _load_and_check(
            TYPE_CKPT, TYPE_META, TARGET_TYPE_ACC, "Stage 1 — Type Ensemble"
        )
    if args.stage in ("grade", "both"):
        results["grade"] = _load_and_check(
            GRADE_CKPT, GRADE_META, TARGET_GRADE_ACC, "Stage 2 — Grade Classifier"
        )

    print("=" * 55)
    all_ok = all(results.values())
    if all_ok:
        print("  ALL CHECKS PASSED — weights are valid and ready.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  SOME CHECKS FAILED: {failed}")
        print("  Re-train the failed stage(s) before using in production.")
    print("=" * 55)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
