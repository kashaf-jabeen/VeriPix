import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from preprocess import train_loader, val_loader

def build_model():
    # 1. Load Pretrained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. Modify last layer (fc) for 2 classes: REAL (1) & FAKE (0)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

def train_model(epochs=5, lr=0.0001, save_path="veripix_model.pth"):
    # 3. Device configuration (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model().to(device)

    # 4. Loss function (Label Smoothing Added) & Optimizer
    # label_smoothing=0.1 model ko 100% overconfident hone se rokeyga
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 6. Save trained model weights
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to {save_path}")

if __name__ == '__main__':
    # Training process start karne ke liye
    train_model(epochs=5)