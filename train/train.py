# train/train.py
import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Simple synthetic dataset for testing (simulates small image classification)
class RandomImageDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=3, channels=3, size=32):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.channels = channels
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # random image-like tensor and a class label
        x = np.random.rand(self.channels, self.size, self.size).astype(np.float32)
        y = np.random.randint(0, self.num_classes)
        return torch.from_numpy(x), y

# Tiny CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def main(args):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # If running in SageMaker, training data will be at /opt/ml/input/data/training
    # But for testing locally we use synthetic data
    if args.use_synthetic:
        train_ds = RandomImageDataset(num_samples=args.samples, num_classes=args.num_classes, size=args.img_size)
        val_ds = RandomImageDataset(num_samples=int(args.samples*0.2), num_classes=args.num_classes, size=args.img_size)
    else:
        # Implement a real dataset loader here (images in class subfolders)
        raise NotImplementedError("Non-synthetic dataset loader not implemented. Use --use_synthetic for testing.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SimpleCNN(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running / len(train_loader)
        train_acc = correct / total
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

        # validation (quick)
        model.eval()
        val_loss = 0.0
        vcorrect = 0
        vtotal = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                vcorrect += (preds == labels).sum().item()
                vtotal += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = vcorrect / vtotal
        print(f"Epoch {epoch}/{args.epochs}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # Save model in SageMaker expected location
    model_dir = args.model_dir or "/opt/ml/model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print("Saved model to:", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=3)
    parser.add_argument("--img-size", dest="img_size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=500)  # synthetic samples
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic random data for testing")
    parser.add_argument("--model-dir", type=str, default=None)
    args = parser.parse_args()
    print("Args:", vars(args))
    main(args)
