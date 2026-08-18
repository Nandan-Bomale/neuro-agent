#!/usr/bin/env python3
"""
download_datasets.py
--------------------
Downloads all additional datasets needed to push model accuracy past 95%.

Run on the DGX server from the neuro-agent root directory:
    python scripts/download_datasets.py

What this downloads:
  1. Brain Tumor MRI Dataset (Kaggle) — 9,257 images for Type classifier
     → Replaces current 5,600 images, adds more Glioma samples
  2. UCSF-PDGM (TCIA) — 501 patients with IDH + MGMT labels
     → 3x more radiogenomics training data
  3. TCGA-GBM + TCGA-LGG (Kaggle subset) — glioma grading data
     → More Grade III / Grade IV slices

Requirements:
    pip install kaggle tcia-utils requests
    Also set up Kaggle API key: https://www.kaggle.com/docs/api
"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path


ROOT = Path(__file__).parent.parent   # neuro-agent/
DATA = ROOT / "data" / "raw"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd: str, cwd: Path = ROOT):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=str(cwd))
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        sys.exit(1)


def check_kaggle():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n" + "="*65)
        print("  Kaggle API key not found!")
        print("  Please set it up:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token' → downloads kaggle.json")
        print("  3. Run: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("  5. Re-run this script")
        print("="*65)
        sys.exit(1)
    print("[✓] Kaggle API key found")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1: Brain Tumor MRI Dataset (Type Classifier)
# Kaggle: masoudnickparvar/brain-tumor-mri-dataset
# 9,257 images across 4 classes (vs current 5,600)
# ─────────────────────────────────────────────────────────────────────────────

def download_tumor_type_dataset():
    dest = DATA / "brain_tumor_mri_v2"
    if dest.exists():
        print(f"[SKIP] {dest} already exists")
        return

    print("\n" + "="*65)
    print("  [1/3] Downloading Brain Tumor MRI Dataset (9,257 images)...")
    print("="*65)

    dest.mkdir(parents=True, exist_ok=True)
    run(f"kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p {dest} --unzip")

    # Confirm structure
    training_dir = dest / "Training"
    testing_dir  = dest / "Testing"
    if training_dir.exists() and testing_dir.exists():
        for cls in training_dir.iterdir():
            n = len(list(cls.glob("*.*")))
            print(f"  {cls.name}: {n} training images")
        print(f"\n[✓] Brain Tumor MRI dataset saved → {dest}")
    else:
        print(f"[WARN] Unexpected folder structure in {dest}. Check manually.")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2: UCSF-PDGM (Radiogenomics - IDH + MGMT)
# 501 patients with full mpMRI + molecular labels
# Available on Kaggle as UCSF-PDGM-v3
# ─────────────────────────────────────────────────────────────────────────────

def download_ucsf_pdgm():
    dest = DATA / "UCSF_PDGM"
    if dest.exists():
        print(f"[SKIP] {dest} already exists")
        return

    print("\n" + "="*65)
    print("  [2/3] Downloading UCSF-PDGM (501 patients, IDH+MGMT)...")
    print("  Note: ~50GB download, may take 30-60 minutes")
    print("="*65)

    dest.mkdir(parents=True, exist_ok=True)

    # Try Kaggle mirror first (faster)
    try:
        run(f"kaggle datasets download -d koussks/ucsf-pdgm-v3 -p {dest} --unzip")
        print(f"[✓] UCSF-PDGM saved → {dest}")
    except SystemExit:
        print("\n[INFO] Kaggle mirror not available. Trying tcia-utils...")
        _download_ucsf_tcia(dest)


def _download_ucsf_tcia(dest: Path):
    """Fallback: download UCSF-PDGM directly from TCIA using tcia-utils."""
    try:
        from tcia_utils import nbia
    except ImportError:
        run("pip install tcia-utils")
        from tcia_utils import nbia

    print("  Fetching series list from TCIA for UCSF-PDGM...")
    series = nbia.getSeries(collection="UCSF-PDGM")
    if not series:
        print("[ERROR] Could not fetch UCSF-PDGM from TCIA. Register at cancerimagingarchive.net")
        return

    print(f"  Found {len(series)} series. Starting download...")
    nbia.downloadSeries(series, path=str(dest), format="nifti")
    print(f"[✓] UCSF-PDGM saved → {dest}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 3: TCGA Glioma Grading (Grade III vs IV)
# Kaggle: awsaf49/brats2020-training-data — has Grade labels
# Also: TCGA-LGG (grade_II/III) from Kaggle
# ─────────────────────────────────────────────────────────────────────────────

def download_tcga_grading():
    dest = DATA / "TCGA_LGG_grading"
    if dest.exists():
        print(f"[SKIP] {dest} already exists")
        return

    print("\n" + "="*65)
    print("  [3/3] Downloading TCGA LGG MRI dataset (grade_II/III)...")
    print("="*65)

    dest.mkdir(parents=True, exist_ok=True)

    # mateuszbuda/lgg-mri-segmentation — 110 patients, FLAIR images + masks
    # Labeled as LGG (Low Grade Glioma = Grade II/III)
    run(f"kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p {dest} --unzip")
    print(f"[✓] TCGA LGG dataset saved → {dest}")

    # Also download glioma grading tabular features (IDH + grade labels)
    dest2 = DATA / "glioma_grading_clinical"
    dest2.mkdir(parents=True, exist_ok=True)
    run(f"kaggle datasets download -d pranavdurai/glioma-grading-clinical-and-mutation-features -p {dest2} --unzip")
    print(f"[✓] Glioma grading features saved → {dest2}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  Neuro Agent — Dataset Downloader")
    print("  Downloads all datasets needed to push accuracy past 95%")
    print("="*65)

    # Install requirements
    run("pip install -q kaggle tcia-utils requests")

    check_kaggle()

    download_tumor_type_dataset()
    download_ucsf_pdgm()
    download_tcga_grading()

    print("\n" + "="*65)
    print("  All downloads complete!")
    print("")
    print("  Next steps:")
    print("  1. Update --data-path for Type to: data/raw/brain_tumor_mri_v2/")
    print("  2. UCSF-PDGM will be integrated into radiogenomics training")
    print("  3. Re-run training commands from README")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
