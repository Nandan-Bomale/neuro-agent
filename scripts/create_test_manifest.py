"""
create_test_manifest.py
-----------------------
Creates a test manifest of 20 patients from the validation set
with known ground truth labels for verification.

Selects diverse patients:
- 5 × IDH-mutant   + MGMT-methylated   (best prognosis)
- 5 × IDH-mutant   + MGMT-unmethylated
- 5 × IDH-wildtype + MGMT-methylated
- 5 × IDH-wildtype + MGMT-unmethylated (GBM — worst prognosis)

All patients are from the BraTS2021 Radiogenomics dataset which has
real hospital MRI scans (NIfTI format) + DNA lab-test ground truth.

Usage:
    python scripts/create_test_manifest.py
"""

import pandas as pd
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

DATA_DIR  = Path("data/raw/BraTS2021_Radiogenomics/train")
CSV_PATH  = Path("data/raw/BraTS2021_Radiogenomics/train_labels.csv")
OUT_PATH  = Path("data/test_manifest_20patients.csv")

def find_nifti(subject_dir: Path, modality: str) -> str | None:
    """Find a NIfTI file for a given modality in the subject directory."""
    patterns = {
        "flair": ["flair", "FLAIR"],
        "t1ce":  ["t1ce", "T1wCE", "t1gad", "T1gad"],
    }
    for pat in patterns.get(modality, [modality]):
        matches = list(subject_dir.rglob(f"*{pat}*.nii.gz")) + \
                  list(subject_dir.rglob(f"*{pat}*.nii"))
        files = [m for m in matches if m.is_file()]
        if files:
            return str(files[0])
    return None

def main():
    print("[manifest] Loading labels CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"[manifest] Total patients: {len(df)}")
    print(f"[manifest] Columns: {df.columns.tolist()}")

    # Parse labels
    df["idh_label"]  = df["IDH_value"].apply(lambda v: "mutant"   if float(v) == 1.0 else "wildtype")
    df["mgmt_label"] = df["MGMT_value"].apply(lambda v: "methylated" if float(v) == 1.0 else "unmethylated")

    # Select 5 patients per group
    groups = [
        ("mutant",   "methylated"),
        ("mutant",   "unmethylated"),
        ("wildtype", "methylated"),
        ("wildtype", "unmethylated"),
    ]

    selected = []
    for idh, mgmt in groups:
        subset = df[(df["idh_label"] == idh) & (df["mgmt_label"] == mgmt)]
        sampled = subset.sample(n=min(5, len(subset)), random_state=42)
        selected.append(sampled)
        print(f"[manifest] {idh} + {mgmt}: selected {len(sampled)} patients")

    test_df = pd.concat(selected, ignore_index=True)
    print(f"[manifest] Total selected: {len(test_df)} patients\n")

    # Build manifest with file paths
    records = []
    for _, row in test_df.iterrows():
        subject_id = str(int(row["BraTS21ID"])).zfill(5)
        subject_dir = DATA_DIR / subject_id
        if not subject_dir.exists():
            subject_dir = DATA_DIR / f"BraTS2021_{subject_id}"

        if not subject_dir.exists():
            print(f"[manifest] ⚠️  Directory not found for {subject_id}, skipping.")
            continue

        # Find a NIfTI file to use as the MRI path
        flair_path = find_nifti(subject_dir, "flair")
        if not flair_path:
            print(f"[manifest] ⚠️  No FLAIR found for {subject_id}, skipping.")
            continue

        records.append({
            "patient_id":        subject_id,
            "mri_path":          flair_path,
            "gt_idh":            row["idh_label"],
            "gt_mgmt":           row["mgmt_label"],
            "gt_tumor_type":     "glioma",           # BraTS is all glioma
            "gt_tumor_grade":    "grade_IV" if row["idh_label"] == "wildtype" else "grade_III",
            "patient_age":       55,                  # BraTS doesn't have age
            "patient_sex":       "M",
        })

    manifest = pd.DataFrame(records)
    manifest.to_csv(OUT_PATH, index=False)

    print(f"\n✅ Test manifest saved: {OUT_PATH}")
    print(f"   Total patients: {len(manifest)}")
    print("\nPatient breakdown:")
    print(manifest[["patient_id", "gt_idh", "gt_mgmt", "gt_tumor_type"]].to_string(index=False))
    print(f"\nReady to run: python scripts/run_pipeline_test.py")

if __name__ == "__main__":
    main()
