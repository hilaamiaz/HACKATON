import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from model import ModelArchitecture


# (You will need to import your DataLoader and Dataset setup here)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelArchitecture().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    # --- YOUR TRAINING LOOP GOES HERE ---
    # for epoch in range(epochs):
    #     for inputs, labels in dataloader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         ...

    print("Training complete. Saving weights...")

    # Required format for saving:
    state_dict = model.cpu().state_dict()
    joblib.dump(state_dict, "weights.joblib")
    print("Saved to weights.joblib")


if __name__ == "__main__":
    train()