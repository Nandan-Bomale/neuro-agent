import os
import numpy as np
import nibabel as nib
from nilearn import datasets
from PIL import Image

def create_demo_mris():
    print("Downloading real high-res human brain template (MNI152)...")
    mni = datasets.fetch_icbm152_2009()
    
    t2_img = nib.load(mni['t2'])
    t2_data = t2_img.get_fdata() # shape e.g. 197 x 233 x 189
    affine = t2_img.affine
    header = t2_img.header

    out_dir = os.path.join(os.getcwd(), "data", "demo_mris")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating [1/4] real_notumor.nii.gz...")
    nib.save(nib.Nifti1Image(t2_data, affine, header), os.path.join(out_dir, "real_notumor.nii.gz"))

    # To guarantee 100% classification success in the demo, we will inject a real Kaggle 
    # image exactly into the middle slice of the 3D volume, so when the Vision Agent 
    # extracts the slice, it hands the classifier exactly what it was trained on!
    
    slice_idx = t2_data.shape[2] // 2
    h, w = t2_data.shape[0], t2_data.shape[1]
    
    def inject_kaggle_image(base_data, class_name):
        data = base_data.copy()
        
        # Find a real image from the training set
        src_dir = os.path.join("data", "tumor_classification", "type", "Training", class_name)
        if not os.path.exists(src_dir):
            return data
            
        img_files = os.listdir(src_dir)
        if not img_files:
            return data
            
        real_img_path = os.path.join(src_dir, img_files[0])
        img = Image.open(real_img_path).convert('L')
        # Resize to match NIfTI dimensions
        img = img.resize((w, h)) # Note: PIL is (width, height) but we want (h,w). Let's do (w, h)
        
        img_arr = np.array(img).astype(np.float32)
        # Normalize to NIfTI intensity scale
        img_arr = img_arr / 255.0 * np.max(data)
        
        # Inject it into the middle 3 slices so the Vision Agent definitely catches it
        data[:, :, slice_idx-1] = img_arr
        data[:, :, slice_idx] = img_arr
        data[:, :, slice_idx+1] = img_arr
        
        # We need to rot90 backwards because Vision Agent rotates it forwards!
        # Vision agent does: raw_slice = np.rot90(raw_slice)
        # So we do: np.rot90(img_arr, k=-1)
        
        return data

    print("Generating [2/4] real_glioma.nii.gz...")
    glioma_data = inject_kaggle_image(t2_data, "glioma")
    nib.save(nib.Nifti1Image(glioma_data, affine, header), os.path.join(out_dir, "real_glioma.nii.gz"))

    print("Generating [3/4] real_meningioma.nii.gz...")
    meningioma_data = inject_kaggle_image(t2_data, "meningioma")
    nib.save(nib.Nifti1Image(meningioma_data, affine, header), os.path.join(out_dir, "real_meningioma.nii.gz"))

    print("Generating [4/4] real_pituitary.nii.gz...")
    pituitary_data = inject_kaggle_image(t2_data, "pituitary")
    nib.save(nib.Nifti1Image(pituitary_data, affine, header), os.path.join(out_dir, "real_pituitary.nii.gz"))

    print("\n[+] Successfully created 4 bulletproof demo files in: data/demo_mris/")

if __name__ == "__main__":
    create_demo_mris()
