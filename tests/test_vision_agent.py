"""
test_vision_agent.py
--------------------
Manual end-to-end test for the Vision Agent.

Run from the project root:
    python tests/test_vision_agent.py

What this does
--------------
  1. Picks 3 BraTS test-split cases (held-out, never seen during training)
  2. Runs VisionAgent.run() on each one
  3. Prints a full result summary for each case
  4. Saves Grad-CAM overlay images to  models/vision/test_outputs/
  5. Prints a final pass/fail summary

Prerequisites
-------------
  - Training must be complete: models/vision/best_model.pth must exist
  - Run from: c:\\Neuro Agent\\
        python tests/test_vision_agent.py
"""

import sys
import os
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from agents.vision_agent.agent import VisionAgent
from agents.vision_agent.dataset import get_test_samples

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT   = "c:/Neuro Agent/models/vision/best_model.pth"
OUTPUT_DIR   = Path("c:/Neuro Agent/models/vision/test_outputs")
N_TEST_CASES = 3        # how many test cases to run (increase for deeper test)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_overlay(result, case_id: str, output_dir: Path) -> None:
    """Save the Grad-CAM overlay and also individual MRI slices for comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4× zoom so the image is large enough to inspect
    overlay_big = cv2.resize(
        result.overlay_image, (0, 0), fx=4, fy=4,
        interpolation=cv2.INTER_NEAREST,
    )
    out_path = output_dir / f"{case_id}_gradcam_overlay.png"
    cv2.imwrite(str(out_path), overlay_big)
    print(f"    Overlay saved : {out_path}")


def print_separator(char="=", width=60):
    print(char * width)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    print_separator()
    print("NeuroAgent -- Vision Agent Manual Test")
    print_separator()

    # Check checkpoint exists
    if not Path(CHECKPOINT).exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
        print("Train the model first: python -m agents.vision_agent.train")
        sys.exit(1)

    # Load test cases (same seed/split as training — truly held-out)
    print(f"\nLoading test split...")
    test_samples = get_test_samples(data_root="c:/Neuro Agent/data/raw")
    print(f"Test split: {len(test_samples)} cases available")
    print(f"Running on first {N_TEST_CASES} cases...\n")

    # Initialise agent once (model loaded once, reused for all cases)
    agent = VisionAgent(checkpoint_path=CHECKPOINT)

    results_summary = []

    for i, sample in enumerate(test_samples[:N_TEST_CASES]):
        # Extract case ID from the flair path
        case_id = Path(sample["flair"]).parent.name
        print_separator("-")
        print(f"Case {i+1}/{N_TEST_CASES} : {case_id}")
        print_separator("-")

        try:
            result = agent.run(
                flair_path = sample["flair"],
                t1_path    = sample["t1"],
                t1ce_path  = sample["t1ce"],
                t2_path    = sample["t2"],
            )

            # Print summary
            print(result.summary())

            # Save overlay
            save_overlay(result, case_id, OUTPUT_DIR)

            results_summary.append({
                "case_id":        case_id,
                "status":         "PASS",
                "tumour":         result.tumour_detected,
                "confidence":     result.confidence_score,
                "volume_cc":      round(result.tumour_volume_voxels / 1000, 1),
                "review":         result.requires_review,
                "time_s":         result.inference_time_s,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary.append({
                "case_id": case_id,
                "status":  "FAIL",
                "error":   str(e),
            })

    # ---------------------------------------------------------------------------
    # Final summary table
    # ---------------------------------------------------------------------------
    print_separator()
    print("FINAL TEST SUMMARY")
    print_separator()
    print(f"{'Case':<35} {'Status':<6} {'Tumour':<8} {'Conf':<6} {'Vol(cc)':<9} {'Review':<8} {'Time(s)'}")
    print("-" * 90)

    all_passed = True
    for r in results_summary:
        if r["status"] == "FAIL":
            all_passed = False
            print(f"{r['case_id']:<35} FAIL   ERROR: {r.get('error','')}")
        else:
            tumour  = "YES" if r["tumour"] else "NO"
            review  = "YES" if r["review"] else "NO"
            print(
                f"{r['case_id']:<35} PASS   "
                f"{tumour:<8} {r['confidence']:.3f}  "
                f"{r['volume_cc']:<9} {review:<8} {r['time_s']:.1f}s"
            )

    print_separator()
    print(f"Overlays saved to: {OUTPUT_DIR}")
    print(f"Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print_separator()


if __name__ == "__main__":
    main()
