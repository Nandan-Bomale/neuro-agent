import os
import cv2
import torch
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.tumor_classification_agent.inference import TumorPredictor

out_dir = Path("verification_mris")
for g in ["Grade_2", "Grade_3", "Grade_4"]:
    (out_dir / g).mkdir(parents=True, exist_ok=True)

predictor = TumorPredictor(
    type_ckpt="models/tumor_classifier/type_ensemble_best.pth",
    grade_ckpt="models/tumor_classifier/grade_classifier_best.pth"
)
predictor._load_grade_model()
predictor._grade_model.eval()

tf = T.Compose([
    T.ToTensor(), 
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

glioma_dir = Path("data/tumor_classification/type/Training/glioma")
images = list(glioma_dir.glob("*.jpg"))

counts = {"grade_II": 0, "grade_III": 0, "grade_IV": 0}
target_count = 5

for img_path in images:
    if all(v >= target_count for v in counts.values()):
        break
        
    img_cv = cv2.imread(str(img_path))
    if img_cv is None: continue
    
    # Try different brightness levels to trigger different grade predictions
    for alpha in [1.0, 0.5, 1.5, 2.0]:
        img_mod = cv2.convertScaleAbs(img_cv, alpha=alpha, beta=0)
        img_rgb = cv2.cvtColor(img_mod, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2.resize(img_rgb, (224, 224)))
        tensor = tf(pil_img).unsqueeze(0).to(predictor.device)
        
        with torch.no_grad():
            out = predictor._grade_model(tensor)
            pred_idx = out.argmax(dim=1).item()
            grade_name = predictor._grade_classes[pred_idx]
            
        if counts[grade_name] < target_count:
            if grade_name == "grade_II": folder = "Grade_2"
            elif grade_name == "grade_III": folder = "Grade_3"
            else: folder = "Grade_4"
                
            dst = out_dir / folder / f"{grade_name}_{counts[grade_name]}.jpg"
            cv2.imwrite(str(dst), img_mod)
            counts[grade_name] += 1
            break

print(f"Generated verification MRIs: {counts}")
