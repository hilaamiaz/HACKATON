import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import joblib
import os
import zipfile
import random
from model import ModelArchitecture


def train():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Path Handling & Auto-Extraction
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_base = os.path.join(script_dir, "dataset")
    zip_path = os.path.join(dataset_base, "train_set.zip")

    # Check if dataset is already extracted (looking for inner folders)
    is_extracted = os.path.exists(dataset_base) and any(
        os.path.isdir(os.path.join(dataset_base, d)) for d in os.listdir(dataset_base)
    )

    if not is_extracted:
        if os.path.exists(zip_path):
            print(f"Found ZIP. Extracting {zip_path} into {dataset_base}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_base)
            print("Extraction complete!")
        else:
            print("Warning: Could not find train_set.zip. Assuming files are manually placed.")

    # 3. Data Augmentation (Robustness Strategy)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # חיתוך אקראי כדי להתעלם מהרקע
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # עיוות צבע
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Dataset & DataLoader Initialization
    full_dataset = datasets.ImageFolder(root=dataset_base, transform=train_transforms)

    DEBUG_MODE = True  # <--- שימי לב: תחליפי ל-False כשרוצים לאמן על כל ה-20,000 תמונות

    if DEBUG_MODE:
        subset_size = int(len(full_dataset) * 0.1)
        subset_indices = random.sample(range(len(full_dataset)), subset_size)
        train_dataset = Subset(full_dataset, subset_indices)
        print(f"DEBUG MODE ON: Training on a 10% subset ({len(train_dataset)} images).")
    else:
        train_dataset = full_dataset
        print(f"Training on full dataset ({len(train_dataset)} images).")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    # 5. Model, Loss, and Optimizer
    model = ModelArchitecture(num_classes=20).to(device)
    criterion = nn.CrossEntropyLoss()
    # שימוש ב-Weight Decay כדי למנוע Overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 6. Training Loop
    epochs = 15  # אפשר להעלות ל-30 או 50 באימון הסופי
    print("\nStarting Training Loop...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 50 == 0:  # הדפסת סטטוס כל 50 באצ'ים
                acc = 100. * correct / total
                print(
                    f"Epoch [{epoch + 1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f} | Acc: {acc:.2f}%")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"==> Epoch {epoch + 1} Summary | Avg Loss: {epoch_loss:.4f} | Avg Acc: {epoch_acc:.2f}% \n")

    # 7. Save Weights Constraints (Move to CPU, save as joblib)
    print("Training finished. Moving model to CPU and saving weights...")
    state_dict = model.cpu().state_dict()

    # שמירה באותה התיקייה שבה נמצא קובץ ה-train.py
    weights_path = os.path.join(script_dir, "weights.joblib")
    joblib.dump(state_dict, weights_path)

    print(f"Success! Model weights saved exactly to: {weights_path}")


if __name__ == "__main__":
    train()