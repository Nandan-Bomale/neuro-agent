"""
test_pipeline.py
----------------
End-to-end pipeline smoke test.

Tests the full Neuro-Agent pipeline with a MOCK MRI path.
All agents that have no checkpoint will gracefully fall back to mock mode.
Agents with real checkpoints (Type, Grade, Radiogenomics) will use real ML.

Usage (run from project root):
    python scripts/test_pipeline.py

    # With a real NIfTI file:
    python scripts/test_pipeline.py --mri path/to/scan.nii.gz

Expected output:
    All 8 nodes execute without crash.
    Final state contains tumor_type, tumor_grade, IDH/MGMT status,
    surgical assessment, survival estimate, and Tumor Board report.
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path

# Fix Windows console emoji encoding issues
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("test_pipeline")


def run_smoke_test(mri_path: str, patient_data: dict) -> None:
    print("\n" + "=" * 70)
    print("  NEURO-AGENT END-TO-END PIPELINE TEST")
    print("=" * 70)
    print(f"  MRI Scan : {mri_path}")
    print(f"  Patient  : Age {patient_data.get('age')} | {patient_data.get('sex', 'M')}")
    print("=" * 70 + "\n")

    try:
        from orchestrator.graph import run_pipeline
    except ImportError as e:
        print(f"[FATAL] Could not import orchestrator: {e}")
        print("Make sure you are running from the project root.")
        sys.exit(1)

    try:
        print("[TEST] Running full pipeline...\n")
        result = run_pipeline(
            mri_slice_path = mri_path,
            patient_data  = patient_data,
        )
    except Exception as e:
        print(f"\n[FATAL] Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — RESULTS")
    print("=" * 70)

    status = result.get("pipeline_status", "unknown")
    print(f"\n✅ Pipeline Status: {status.upper()}")

    # Vision Agent
    vf = result.get("vision_findings", {})
    print(f"\n📍 Vision Agent:")
    print(f"   Tumor Detected : {vf.get('tumour_detected')}")
    print(f"   Volume         : {vf.get('tumour_volume_cc')} cc")
    print(f"   Confidence     : {vf.get('confidence_score')}")

    # Classification Agent
    clf = result.get("tumor_classification_findings", {})
    print(f"\n🔬 Classification Agent:")
    print(f"   Tumor Type    : {clf.get('tumor_type')}")
    print(f"   Tumor Grade   : {clf.get('tumor_grade')}")
    print(f"   Urgency       : {clf.get('clinical_urgency')}")
    print(f"   Confidence    : {clf.get('confidence')}")

    # Radiogenomics Agent
    radio = result.get("radiogenomics_findings", {})
    print(f"\n🧬 Radiogenomics Agent:")
    print(f"   IDH Mutation  : {radio.get('idh_mutation_status')} (conf={radio.get('idh_confidence')})")
    print(f"   MGMT Status   : {radio.get('mgmt_methylation_status')} (conf={radio.get('mgmt_confidence')})")
    print(f"   Source        : {radio.get('source')}")

    # Surgical Agent
    surg = result.get("surgical_analysis", {})
    print(f"\n🔪 Surgical Agent:")
    print(f"   Resectability : {int((surg.get('resectability_score', 0)) * 100)}%")
    print(f"   Eloquent Area : {surg.get('eloquent_area_proximity')}")
    print(f"   Recommendation: {surg.get('surgical_recommendation')}")

    # Prognostic Agent
    prog = result.get("prognostic_analysis", {})
    print(f"\n📊 Prognostic Agent:")
    print(f"   Overall Survival     : {prog.get('overall_survival_months')} months")
    print(f"   Progression-Free     : {prog.get('progression_free_survival_months')} months")
    print(f"   Risk Category        : {prog.get('risk_category', '').upper()}")

    # Clinical Trials
    trials = result.get("clinical_trials", [])
    print(f"\n🔍 Clinical Trial Agent:")
    print(f"   Trials Found: {len(trials)}")
    if trials:
        print(f"   Example: {trials[0].get('nctId', 'N/A')} — {trials[0].get('briefTitle', '')[:60]}")

    # Neuro-Oncologist
    plan = result.get("neuro_oncologist_plan", {})
    print(f"\n💊 Neuro-Oncologist Agent:")
    print(f"   Protocol      : {plan.get('chemotherapy_protocol')}")
    print(f"   Radiotherapy  : {plan.get('radiotherapy_protocol')}")
    rec = plan.get("treatment_recommendation", "")
    if rec:
        print(f"   Recommendation: {rec[:120]}...")

    # Explainability — Final Report
    explanation = result.get("explanation_summary", "")
    if explanation:
        print(f"\n{explanation}")

    # Verification
    print(f"\n🔒 Verification Status: {result.get('verification_status', '?').upper()}")
    if result.get("human_review_reason"):
        print(f"   Reason: {result.get('human_review_reason')}")

    print("\n" + "=" * 70)
    print("  ALL AGENTS EXECUTED SUCCESSFULLY!")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Neuro-Agent end-to-end pipeline test")
    parser.add_argument(
        "--mri",
        default="data/raw/BraTS2021_Radiogenomics/train/00000/FLAIR.nii.gz",
        help="Path to the NIfTI MRI scan file",
    )
    parser.add_argument("--age",  type=int,   default=55,  help="Patient age")
    parser.add_argument("--sex",  type=str,   default="M", help="Patient sex (M/F)")
    args = parser.parse_args()

    patient_data = {
        "age":       args.age,
        "sex":       args.sex,
        "symptoms":  ["headache", "seizures"],
        "history":   "Patient presenting with new-onset seizures and progressive headaches over 3 months.",
    }

    run_smoke_test(mri_path=args.mri, patient_data=patient_data)


if __name__ == "__main__":
    main()
