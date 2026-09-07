import sys, cv2, numpy as np
sys.path.insert(0, '.')
from ultralytics import YOLO
from pathlib import Path

model = YOLO('models/yolo/weights/detect_best.pt')
img_path = 'C:/Users/geeta/.gemini/antigravity/brain/95005061-86b6-4fa5-b84d-6dd844c76df5/.user_uploaded/media_1787647798230.png'

img = cv2.imread(img_path)
if img is None:
    print('Cannot read image'); sys.exit(1)

H, W = img.shape[:2]
print(f'Image size: {W}x{H}')

results = model(img, conf=0.05, verbose=False, iou=0.3, device='cpu')
r = results[0]
print(f'Total raw detections: {len(r.boxes) if r.boxes else 0}')
if r.boxes:
    for i, box in enumerate(r.boxes):
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        cls_name = model.names.get(cls, '?')
        cx, cy = (x1+x2)//2, (y1+y2)//2
        # What fraction down the image is this?
        frac = cy / H
        print(f'  [{i}] cls={cls_name} conf={conf:.3f} box=({x1},{y1},{x2-x1},{y2-y1}) center_y={cy} ({frac*100:.0f}% down)')
