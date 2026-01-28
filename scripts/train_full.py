import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models

from dataset import DeepfakeDataset
from preprocessing import train_transform, val_transform

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 3)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_dataset = DeepfakeDataset("data", transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

epochs = 2
for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "models/baseline_resnet18.pth")
print("Model saved")
