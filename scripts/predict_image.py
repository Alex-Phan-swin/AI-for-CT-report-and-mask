from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_ROOT / "demo_input"
MODEL_PATH = PROJECT_ROOT / "liver_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]

    if predicted_class == "Hepatic_Steatosis":
        predicted_class = "Unhealthy"

    return predicted_class

valid_extensions = (".png", ".jpg", ".jpeg")
files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]

if not files:
    print(f"No image files found in {INPUT_DIR}")
else:
    print("Predictions:")
    for file_path in sorted(files):
        prediction = predict_image(file_path)
        print(f"{file_path.name}: {prediction}")