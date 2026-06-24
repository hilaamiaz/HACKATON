import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import joblib
import os
import zipfile  # <-- הוספנו את הספרייה הזו

from model import ModelArchitecture


def train():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- 1.5 חילוץ אוטומטי של קובץ ה-ZIP ---
    dataset_path = os.path.join("..", "dataset", "train_set")
    zip_path = "train_set.zip"  # מניח שקובץ ה-ZIP נמצא באותה תיקייה של train.py

    # אם התיקייה עוד לא קיימת, נחלץ אותה מה-ZIP
    if not os.path.exists(dataset_path):
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path} to {dataset_path}...")
            # יוצר את תיקיית היעד במקרה שהיא לא קיימת
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join("..", "dataset"))
            print("Extraction complete!")
        else:
            print(f"⚠️ Warning: Could not find zip file at {zip_path} and dataset folder does not exist.")

    # 1. Data Augmentation for Robustness
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation must be clean (no random augmentations)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Dataset and Split Properly
    # (הקוד כבר משתמש ב-dataset_path שהגדרנו למעלה)
    full_train_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transforms)
    full_val_dataset = datasets.ImageFolder(root=dataset_path, transform=val_transforms)

    # Generate random indices for splitting
    num_samples = len(full_train_dataset)
    indices = torch.randperm(num_samples).tolist()
    train_size = int(0.75 * num_samples)

    # Create Subsets using the randomized indices
    train_dataset = Subset(full_train_dataset, indices[:train_size])
    val_dataset = Subset(full_val_dataset, indices[train_size:])

    # DataLoaders (num_workers=0 is safer on Windows to prevent freezes)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 3. Initialize Model, Loss, and Optimizer
    model = ModelArchitecture().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss / len(train_loader):.4f} | Val Accuracy: {val_accuracy:.2f}%")

    # 5. Save the exact format required
    print("Training complete. Saving weights...")
    state_dict = model.cpu().state_dict()
    joblib.dump(state_dict, "weights.joblib")
    print("Saved to weights.joblib")


if __name__ == "__main__":
    train()