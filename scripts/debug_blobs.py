"""Debug: print ALL candidates for each image to understand which blobs compete."""
import sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Copy the internal logic inline so we can see all candidates
from agents.vision_2d_agent.agent import _get_brain_mask, _enumerate_blobs, _cam_overlap, Vision2DAgent

agent = Vision2DAgent()
agent._init_predictor()

for imgname in [
    "data/interim/preprocessing/preprocessed_1787547244.jpg",  # coronal - tumor upper-center
    "data/interim/preprocessing/preprocessed_1787547063.jpg",  # axial ring - ring left-lower
    "data/interim/preprocessing/preprocessed_1787547103.jpg",  # BraTS dark
]:
    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
    cam = agent._compute_cam(img)
    bm, bx, by, bw, bh, cx, cy = _get_brain_mask(img)
    blobs = _enumerate_blobs(img, bm, cx, cy, bw, bh)
    
    print(f"\n{'='*60}")
    print(f"IMAGE: {imgname}")
    print(f"Brain: x={bx},y={by},w={bw},h={bh}, centroid=({cx},{cy})")
    print(f"CAM max: {cam.max():.3f}, CAM mean: {cam.mean():.3f}")
    print(f"Total blobs found: {len(blobs)}")
    
    # Score them like _fuse does
    cam_max = float(cam.max())
    cam_strong = cam_max > 0.4
    
    for b in blobs:
        x, y, w, h = b["box"]
        bcx, bcy = x+w/2, y+h/2
        iy = int(np.clip(bcy, 0, cam.shape[0]-1))
        ix = int(np.clip(bcx, 0, cam.shape[1]-1))
        cam_at_c = float(cam[iy, ix])
        overlap = _cam_overlap(cam, b["box"])
        if cam_strong:
            if cam_at_c > 0.35 or overlap > 0.30:
                cf = 1.0 + 3.0 * max(cam_at_c, overlap)
            else:
                cf = 0.25
        else:
            cf = 1.0
        final = b["score"] * cf
        print(f"  box=({x},{y},{w},{h}) type={b['blob_type']} anom={b['anomaly']:.2f} "
              f"raw_score={b['score']:.1f} cam_c={cam_at_c:.2f} overlap={overlap:.2f} cf={cf:.2f} final={final:.1f}")
