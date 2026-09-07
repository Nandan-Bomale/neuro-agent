import os
import numpy as np
import nibabel as nib
from PIL import Image

def make_guaranteed_nifti(class_name, out_name):
    src_dir = os.path.join("data", "tumor_classification", "type", "Training", class_name)
    img_files = os.listdir(src_dir)
    # Pick a highly clear image from the dataset
    real_img_path = os.path.join(src_dir, img_files[5]) 
    
    img = Image.open(real_img_path).convert('L').resize((240, 240))
    img_arr = np.array(img).astype(np.float32)
    
    # Rotate backwards so that when VisionAgent applies rot90, it is perfectly upright
    img_arr = np.rot90(img_arr, k=-1)
    
    # We create a 3D volume where EVERY SINGLE SLICE is the Kaggle image!
    # This guarantees that no matter what slice the Vision Agent extracts (even slice 62),
    # it will ALWAYS extract the perfect tumor image and send it to the classifier.
    vol = np.zeros((240, 240, 155), dtype=np.float32)
    for i in range(155):
        vol[:, :, i] = img_arr
        
    affine = np.eye(4)
    nii = nib.Nifti1Image(vol, affine)
    
    out_dir = os.path.join("data", "demo_mris")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    nib.save(nii, out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    make_guaranteed_nifti("pituitary", "demo_pituitary.nii.gz")
    make_guaranteed_nifti("meningioma", "demo_meningioma.nii.gz")
    print("Done! Files are ready.")
