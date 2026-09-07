import cv2
import numpy as np
import os
from pathlib import Path

# Show stats for all recent preprocessed images
imgs_dir = Path("data/interim/preprocessing")
recent = sorted(imgs_dir.glob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)[:8]

for p in recent:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    # Brain mask - any non-black pixel
    brain = img[img > 10]
    if len(brain) == 0: continue
    print(f"{p.name}: shape={img.shape}, brain_mean={brain.mean():.1f}, brain_max={brain.max()}, p90={np.percentile(brain, 90):.1f}, p99={np.percentile(brain, 99):.1f}")
