import sys, cv2, numpy as np
sys.path.insert(0, '.')
from ultralytics import YOLO
from agents.vision_2d_agent.agent import _get_brain_mask

model = YOLO('models/yolo/weights/detect_best.pt')
img_path = 'C:/Users/geeta/.gemini/antigravity/brain/95005061-86b6-4fa5-b84d-6dd844c76df5/.user_uploaded/media_1787647798230.png'

img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = img.shape[:2]

brain_mask, bx, by, bw, bh, cx, cy = _get_brain_mask(img_gray)
brain_pixels = img_gray[brain_mask > 0].astype(float)
brain_mean = float(brain_pixels.mean())
brain_std  = float(brain_pixels.std())

brain_ys = np.where(brain_mask > 0)[0]
brain_top_y = int(brain_ys.min())
brain_bot_y = int(brain_ys.max())
brain_mid_y = (brain_top_y + brain_bot_y) / 2
skull_base_y = brain_top_y + int((brain_bot_y - brain_top_y) * 0.80)

print(f'Image: {W}x{H}')
print(f'Brain: top={brain_top_y} bot={brain_bot_y} mid={brain_mid_y:.0f} skull_base_y={skull_base_y}')
print(f'Brain mean={brain_mean:.1f} std={brain_std:.1f}')
print()

results = model(img, conf=0.05, verbose=False, iou=0.3, device='cpu')
r = results[0]
print(f'Detections: {len(r.boxes)}')
for i, box in enumerate(r.boxes):
    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
    conf = float(box.conf[0])
    cls_name = model.names.get(int(box.cls[0]), '?')
    bcx, bcy = (x1+x2)//2, (y1+y2)//2
    in_brain = brain_mask[min(bcy,H-1), min(bcx,W-1)] > 0
    
    roi = img_gray[max(0,y1):min(H,y2), max(0,x1):min(W,x2)].astype(float)
    roi_mean = float(roi.mean()) if roi.size > 0 else 0
    anom = abs(roi_mean - brain_mean) / max(brain_std, 1.0)
    anom_norm = min(anom / 5.0, 1.0)
    
    upper_bonus = 1.5 if bcy < brain_mid_y else 1.0
    skull_pen   = 0.5 if bcy > skull_base_y else 1.0
    combined    = (conf*0.55 + anom_norm*0.45) * skull_pen * upper_bonus
    
    print(f'  [{i}] {cls_name} conf={conf:.3f} bcy={bcy} ({bcy/H*100:.0f}%) in_brain={in_brain} '
          f'anom={anom:.2f} skull={bcy>skull_base_y} upper={bcy<brain_mid_y} combined={combined:.3f}')
