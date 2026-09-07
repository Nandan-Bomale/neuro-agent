import sys, cv2, numpy as np
sys.path.insert(0, '.')
from agents.vision_2d_agent.agent import _get_brain_mask, _merge_boxes
from ultralytics import YOLO

img_path = 'C:/Users/geeta/.gemini/antigravity/brain/95005061-86b6-4fa5-b84d-6dd844c76df5/.user_uploaded/media_1787647798230.png'
model = YOLO('models/yolo/weights/detect_best.pt')
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = img.shape[:2]
brain_mask, bx, by, bw, bh, cx, cy = _get_brain_mask(img_gray)
brain_pixels = img_gray[brain_mask > 0].astype(float)
brain_mean = float(brain_pixels.mean())
brain_std  = max(float(brain_pixels.std()), 1.0)
brain_ys = np.where(brain_mask > 0)[0]
brain_top_y  = int(brain_ys.min())
brain_bot_y  = int(brain_ys.max())
brain_mid_y  = (brain_top_y + brain_bot_y) / 2
skull_base_y = brain_top_y + int((brain_bot_y - brain_top_y) * 0.80)

results = model(img, conf=0.05, verbose=False, iou=0.3, device='cpu')
r = results[0]

candidates = []
for i, box in enumerate(r.boxes):
    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
    x1,y1 = max(0,x1),max(0,y1); x2,y2 = min(W,x2),min(H,y2)
    conf = float(box.conf[0])
    cls_name = model.names.get(int(box.cls[0]), '?')
    if cls_name == 'negative': continue
    bcx = min((x1+x2)//2, W-1); bcy = min((y1+y2)//2, H-1)
    if brain_mask[bcy, bcx] == 0: continue
    roi = img_gray[y1:y2, x1:x2].astype(float)
    anom = abs(float(roi.mean()) - brain_mean) / brain_std
    upper_bonus = 1.5 if bcy < brain_mid_y else 1.0
    skull_pen   = 0.5 if bcy > skull_base_y else 1.0
    combined    = (conf*0.55 + min(anom/5.0,1.0)*0.45) * skull_pen * upper_bonus
    skull_base  = bcy > skull_base_y
    candidates.append({'box':(x1,y1,x2-x1,y2-y1),'conf':conf,'anom':anom,'combined':combined,'skull_base':skull_base,'bcy':bcy})
    print(f'  [{i}] conf={conf:.3f} bcy={bcy} skull={skull_base} upper={bcy<brain_mid_y} anom={anom:.2f} combined={combined:.3f}')

candidates.sort(key=lambda c:c['combined'], reverse=True)
non_skull = [c for c in candidates if not c['skull_base']]
use_cands = non_skull if non_skull else candidates
thresh = use_cands[0]['combined'] * 0.60
top_k = [c for c in use_cands if c['combined'] >= thresh]
print(f'Top-k boxes: {[c[\"box\"] for c in top_k]}')
merged, mconf = _merge_boxes([(c['box'],c['conf']) for c in top_k], (H,W))
print(f'MERGED BOX: {merged}')
