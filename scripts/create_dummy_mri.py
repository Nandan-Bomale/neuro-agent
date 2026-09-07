import os
import numpy as np
import nibabel as nib

def create_dummy_mri(output_path="data/sample_mri.nii.gz"):
    print("Generating synthetic 3D MRI scan...")
    # Create a 3D volume (128x128x64)
    volume = np.zeros((128, 128, 64), dtype=np.float32)
    
    # Draw a "brain" (a large ellipsoid with moderate intensity)
    for x in range(128):
        for y in range(128):
            for z in range(64):
                if ((x - 64)**2 / 45**2) + ((y - 64)**2 / 55**2) + ((z - 32)**2 / 25**2) <= 1:
                    volume[x, y, z] = 0.4  # brain tissue intensity
                    
    # Draw a "tumor" (a small, highly intense sphere inside the brain)
    for x in range(128):
        for y in range(128):
            for z in range(64):
                if ((x - 40)**2 + (y - 80)**2 + (z - 35)**2) <= 12**2:
                    volume[x, y, z] = 1.0  # tumor intensity (bright)

    # Convert to NIfTI format and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(img, output_path)
    print(f"✅ Synthetic MRI saved to: {output_path}")

if __name__ == "__main__":
    create_dummy_mri()
