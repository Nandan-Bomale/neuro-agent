import torch
import time
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.tumor_classification_agent.inference import TumorPredictor

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    test_dir = Path("data/tumor_classification/type/Testing")
    dataset = datasets.ImageFolder(test_dir, transform=tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    print("Loading models...")
    predictor = TumorPredictor(
        type_ckpt="models/tumor_classifier/type_ensemble_best.pth",
        grade_ckpt="models/tumor_classifier/grade_classifier_best.pth"
    )
    predictor._load_type_model()
    predictor._type_model.eval()

    correct = 0
    total = 0

    print("Evaluating Type Model on Test MRIs (No TTA)...")
    t0 = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = predictor._type_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Accuracy of the Type Model on the {total} test images: {acc:.2f}%")
    print(f"Evaluation took {time.time()-t0:.2f}s")
