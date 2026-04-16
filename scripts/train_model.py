from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
from Loading_Dataset import LiverDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "liver_model.pth"

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Dataset location:", DATASET_DIR)

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# Load dataset
# -----------------------------
dataset = LiverDataset(root_dir=DATASET_DIR, transform=transform)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

if len(dataset.classes) < 2:
    print("Error: need at least 2 classes in dataset/")
    raise SystemExit(1)

# -----------------------------
# DataLoader
# -----------------------------
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# -----------------------------
# Model
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))
model = model.to(device)

# -----------------------------
# Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training
# -----------------------------
num_epochs = 6

train_accuracies = []
train_losses = []
train_precisions = []
train_recalls = []
train_f1s = []

for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # store for precision/recall/F1
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix(loss=loss.item())

    # epoch metrics
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    train_accuracies.append(accuracy)
    train_losses.append(running_loss)
    train_precisions.append(precision)
    train_recalls.append(recall)
    train_f1s.append(f1)

    print(f"""
Epoch {epoch+1} Results:
Loss: {running_loss:.4f}
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}
""")

epochs = range(1, num_epochs + 1)

plt.plot(epochs, train_accuracies, label="Accuracy")
plt.plot(epochs, train_precisions, label="Precision")
plt.plot(epochs, train_recalls, label="Recall")
plt.plot(epochs, train_f1s, label="F1-score")

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Training Metrics")
plt.legend()
plt.show()

# -----------------------------
# Save model
# -----------------------------
MODELS_DIR.mkdir(exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": dataset.classes
}, MODEL_PATH)

print(f"Model saved as {MODEL_PATH}")