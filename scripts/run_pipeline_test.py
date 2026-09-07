"""
run_pipeline_test.py
--------------------
Runs the full Neuro-Agent pipeline on 20 test patients and generates
a verification report comparing AI predictions vs ground truth labels.

Usage (on DGX):
    python scripts/run_pipeline_test.py

Requires:
    - data/test_manifest_20patients.csv  (created by create_test_manifest.py)
    - models/radiogenomics/radiogenomics_best.pth
    - models/tumor_classifier/type_ensemble_best.pth
    - models/tumor_classifier/grade_classifier_best.pth

Output:
    - Prints a full comparison table (AI prediction vs Ground Truth)
    - Saves results to: data/pipeline_test_results.csv
"""

import sys
import os
import logging
import time
import pandas as pd
from pathlib import Path

# Fix Windows console emoji encoding issues
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level   = logging.WARNING,   # Only show warnings during batch test
    format  = "%(levelname)s | %(name)s | %(message)s",
)

MANIFEST_PATH = Path("data/test_manifest_20patients.csv")
RESULTS_PATH  = Path("data/pipeline_test_results.csv")

PASS  = "✅"
FAIL  = "❌"
SKIP  = "⚠️"


def run_single_patient(row: dict) -> dict:
    """Run the pipeline on a single patient and return results."""
    from orchestrator.graph import run_pipeline

    patient_data = {
        "age":      row.get("patient_age", 55),
        "sex":      row.get("patient_sex", "M"),
        "symptoms": ["headache"],
        "history":  "Test patient from BraTS2021 Radiogenomics dataset.",
    }

    try:
        state = run_pipeline(
            mri_scan_path = row["mri_path"],
            patient_data  = patient_data,
        )

        clf   = state.get("tumor_classification_findings", {})
        radio = state.get("radiogenomics_findings", {})
        surg  = state.get("surgical_analysis", {})
        prog  = state.get("prognostic_analysis", {})

        return {
            "patient_id":       row["patient_id"],
            "status":           state.get("pipeline_status", "error"),
            # Predictions
            "pred_tumor_type":  clf.get("tumor_type"),
            "pred_tumor_grade": clf.get("tumor_grade"),
            "pred_idh":         radio.get("idh_mutation_status"),
            "pred_mgmt":        radio.get("mgmt_methylation_status"),
            "pred_idh_conf":    radio.get("idh_confidence"),
            "pred_mgmt_conf":   radio.get("mgmt_confidence"),
            "resectability":    surg.get("resectability_score"),
            "os_months":        prog.get("overall_survival_months"),
            "risk":             prog.get("risk_category"),
            # Ground truth
            "gt_tumor_type":    row["gt_tumor_type"],
            "gt_tumor_grade":   row["gt_tumor_grade"],
            "gt_idh":           row["gt_idh"],
            "gt_mgmt":          row["gt_mgmt"],
            # Correctness flags
            "idh_correct":      radio.get("idh_mutation_status") == row["gt_idh"],
            "mgmt_correct":     radio.get("mgmt_methylation_status") == row["gt_mgmt"],
            "type_correct":     clf.get("tumor_type") == row["gt_tumor_type"],
            "error":            None,
        }

    except Exception as e:
        return {
            "patient_id": row["patient_id"],
            "status":     "error",
            "error":      str(e),
            "idh_correct": False,
            "mgmt_correct": False,
            "type_correct": False,
        }


def print_results_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("  NEURO-AGENT PIPELINE VERIFICATION REPORT — 20 PATIENT BATCH TEST")
    print("=" * 90)

    print(f"\n{'PID':<8} {'GT IDH':<12} {'PRED IDH':<14} {'IDH?':<6} "
          f"{'GT MGMT':<14} {'PRED MGMT':<16} {'MGMT?':<6} "
          f"{'OS (mo)':<10} {'RISK':<8}")
    print("-" * 90)

    idh_correct  = 0
    mgmt_correct = 0
    type_correct = 0
    valid        = 0

    for r in results:
        if r.get("status") == "error":
            print(f"{r['patient_id']:<8} ERROR: {r.get('error', '')[:60]}")
            continue

        valid += 1
        idh_ok  = r.get("idh_correct",  False)
        mgmt_ok = r.get("mgmt_correct", False)
        type_ok = r.get("type_correct", False)

        if idh_ok:  idh_correct  += 1
        if mgmt_ok: mgmt_correct += 1
        if type_ok: type_correct += 1

        print(
            f"{r['patient_id']:<8} "
            f"{r.get('gt_idh','?'):<12} {r.get('pred_idh','?'):<14} "
            f"{PASS if idh_ok else FAIL:<6} "
            f"{r.get('gt_mgmt','?'):<14} {r.get('pred_mgmt','?'):<16} "
            f"{PASS if mgmt_ok else FAIL:<6} "
            f"{str(r.get('os_months','?')):<10} "
            f"{str(r.get('risk','?')):<8}"
        )

    if valid > 0:
        idh_acc  = idh_correct  / valid * 100
        mgmt_acc = mgmt_correct / valid * 100
        type_acc = type_correct / valid * 100

        print("\n" + "=" * 90)
        print("  ACCURACY SUMMARY")
        print("=" * 90)
        print(f"  IDH  Mutation Prediction  : {idh_correct}/{valid}  = {idh_acc:.1f}%")
        print(f"  MGMT Methylation Prediction: {mgmt_correct}/{valid} = {mgmt_acc:.1f}%")
        print(f"  Tumor Type Prediction      : {type_correct}/{valid}  = {type_acc:.1f}%")
        print("=" * 90)


def main():
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Test manifest not found: {MANIFEST_PATH}")
        print("Run first: python scripts/create_test_manifest.py")
        sys.exit(1)

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"\n[TEST] Running pipeline on {len(manifest)} patients...")
    print("       (This will take a few minutes)\n")

    results = []
    for i, (_, row) in enumerate(manifest.iterrows()):
        pid = row["patient_id"]
        print(f"  [{i+1:02d}/{len(manifest)}] Patient {pid}...", end=" ", flush=True)
        t0 = time.time()
        r  = run_single_patient(row.to_dict())
        elapsed = time.time() - t0
        status = "DONE" if r.get("status") != "error" else "ERROR"
        print(f"{status} ({elapsed:.1f}s)")
        results.append(r)

    # Save results
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\n[TEST] Results saved: {RESULTS_PATH}")

    # Print formatted report
    print_results_table(results)


if __name__ == "__main__":
    main()
