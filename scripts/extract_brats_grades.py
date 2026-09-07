import os
import cv2
import nibabel as nib
import numpy as np
from pathlib import Path
import random

out_dir = Path("verification_mris")
for g in ["Grade_3", "Grade_4"]:
    (out_dir / g).mkdir(parents=True, exist_ok=True)

brats_dir = Path("data/raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
patient_dirs = [d for d in brats_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS20_Training_")]

random.seed(42)
random.shuffle(patient_dirs)

counts = {"Grade_3": 0, "Grade_4": 0}
target = 5

for pdir in patient_dirs:
    if counts["Grade_3"] >= target and counts["Grade_4"] >= target:
        break
        
    seg_file = pdir / f"{pdir.name}_seg.nii"
    t1ce_file = pdir / f"{pdir.name}_t1ce.nii"
    
    if not seg_file.exists() or not t1ce_file.exists():
        continue
        
    try:
        seg_data = nib.load(str(seg_file)).get_fdata()
        # In BraTS 2020, label 4 is Enhancing Tumor (ET)
        et_pixels = np.sum(seg_data == 4)
        
        if et_pixels < 50:
            continue # Grade 2, skip
        elif et_pixels < 3000:
            grade = "Grade_3"
        else:
            grade = "Grade_4"
            
        if counts[grade] < target:
            # find the slice with the most ET
            et_per_slice = [np.sum(seg_data[:,:,z] == 4) for z in range(seg_data.shape[2])]
            best_z = np.argmax(et_per_slice)
            
            # Load T1ce
            t1ce_data = nib.load(str(t1ce_file)).get_fdata()
            slice_img = t1ce_data[:, :, best_z]
            
            # Normalize and convert to uint8
            if slice_img.max() > 0:
                slice_img = (slice_img / slice_img.max() * 255).astype(np.uint8)
                
            # Rotate correctly
            slice_img = np.rot90(slice_img, k=-1)
            
            out_path = out_dir / grade / f"{pdir.name}_slice{best_z}.jpg"
            cv2.imwrite(str(out_path), slice_img)
            counts[grade] += 1
            print(f"Saved {grade} image: {out_path} (ET pixels: {et_pixels})")
            
    except Exception as e:
        print(f"Error processing {pdir.name}: {e}")

print(f"Final counts: {counts}")
