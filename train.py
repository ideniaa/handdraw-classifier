import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from model import SketchCNN, class_labels

# Load QuickDraw dataset
class QuickDrawDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # FIX: Use class_labels (fixed order from model.py) instead of os.listdir()
        # which returns an arbitrary OS-dependent order. Using a different order here
        # than in drawing_app.py causes every prediction to map to the wrong label.
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_labels)}

        for class_name in class_labels:
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                print(f"Warning: directory not found for class '{class_name}', skipping.")
                continue
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Load dataset
dataset = QuickDrawDataset("quickdraw_data", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_labels)
model = SketchCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
epochs = 10
for epoch in range(epochs):
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "quickdraw_model.pth")
print("Model training complete!")
