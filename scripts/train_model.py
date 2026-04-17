from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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
# Train / Validation split
# -----------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

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
num_epochs = 30

train_losses = []
train_accuracies = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []

for epoch in range(num_epochs):

    # =====================
    # TRAINING
    # =====================
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")

    for images, labels in train_bar:
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

        train_bar.set_postfix(loss=loss.item())

    train_acc = correct / total
    train_losses.append(running_loss)
    train_accuracies.append(train_acc)

    # =====================
    # VALIDATION
    # =====================
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]")

        for images, labels in val_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = (sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels))
    val_precision = precision_score(all_labels, all_preds, average='weighted')
    val_recall = recall_score(all_labels, all_preds, average='weighted')
    val_f1 = f1_score(all_labels, all_preds, average='weighted')

    val_accuracies.append(val_acc)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    val_f1s.append(val_f1)

    # =====================
    # PRINT RESULTS
    # =====================
    print(f"""
Epoch {epoch+1} Results:
------------------------
Train Loss: {running_loss:.4f}
Train Accuracy: {train_acc:.4f}

VAL Accuracy: {val_acc:.4f}
VAL Precision: {val_precision:.4f}
VAL Recall: {val_recall:.4f}
VAL F1-score: {val_f1:.4f}
""")

# -----------------------------
# Plot metrics
# -----------------------------
epochs = range(1, num_epochs + 1)

plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies, label="Val Accuracy")
plt.plot(epochs, val_f1s, label="Val F1-score")

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Training vs Validation Metrics")
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