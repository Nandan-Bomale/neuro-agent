import os
import numpy as np
import nibabel as nib
from PIL import Image

def build_massive_demo_dataset():
    src_base = os.path.join("data", "tumor_classification", "type", "Testing")
    out_base = os.path.join("data", "Demo_Testing_Dataset_All_Tumors")
    
    classes = ["glioma", "meningioma", "pituitary", "notumor"]
    
    for cls in classes:
        src_dir = os.path.join(src_base, cls)
        out_dir = os.path.join(out_base, cls)
        os.makedirs(out_dir, exist_ok=True)
        
        if not os.path.exists(src_dir):
            continue
            
        img_files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
        # Take the first 10 images from each class (40 total)
        for img_file in img_files[:10]:
            real_img_path = os.path.join(src_dir, img_file)
            
            img = Image.open(real_img_path).convert('L').resize((240, 240))
            img_arr = np.array(img).astype(np.float32)
            img_arr = np.rot90(img_arr, k=-1) # Align for the Vision Agent
            
            vol = np.zeros((240, 240, 155), dtype=np.float32)
            for i in range(155):
                vol[:, :, i] = img_arr
                
            affine = np.eye(4)
            nii = nib.Nifti1Image(vol, affine)
            
            out_name = f"patient_{cls}_{img_file.replace('.jpg', '.nii.gz')}"
            out_path = os.path.join(out_dir, out_name)
            nib.save(nii, out_path)
            print(f"Created {out_path}")

if __name__ == "__main__":
    build_massive_demo_dataset()
    print("ALL FILES GENERATED SUCCESSFULLY.")
