import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# ---
# Challenge : Limited and Imbalanced Data
# Solution : Apply aggressive data augmentation (random flips, resizing, normalization)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),                     # Ensures all images are the same input size for the model
    transforms.RandomHorizontalFlip(),                 # Adds diversity to the dataset to prevent overfitting
    transforms.ToTensor(),                             # Converts images from PIL to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406],        # Normalizes images; matches pretrained model expectations
                     [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),                     # Validation and test images resized for consistency
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])
])

# Solution : Use PyTorch's ImageFolder and DataLoader for scalable, reproducible training
train_dataset = datasets.ImageFolder('data_split/train', transform=train_transforms)
val_dataset = datasets.ImageFolder('data_split/val', transform=val_test_transforms)
test_dataset = datasets.ImageFolder('data_split/test', transform=val_test_transforms)

# Create DataLoader objects for batching and epoch shuffling (critical for generalization)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---
# Challenge : Transfer Learning & Model Adaptation
# Solution : Load a ResNet50 pretrained on ImageNet, then freeze its backbone weights to preserve general visual features
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False    # Only train the classifier head for efficiency and to reduce risk of overfitting

# Replace final classifier layer with a custom head for binary classification (Covid/Normal)
num_classes = 2
model.fc = nn.Sequential(
    nn.Dropout(),                 # Regularization: Dropout prevents overfitting (Challenge 2)
    nn.Linear(model.fc.in_features, num_classes)
)

# ---
# Challenge : Efficient Hardware Utilization
# Solution: Move model to GPU if available, for fast training and experimentation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ---
# Challenge: Imbalance and Overfitting
# Solution: Use Cross Entropy Loss (optionally weighted for class imbalance); Optimize only classifier head
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# ---
# Challenge: Robust Training & Early Stopping
# Solution: Track validation metrics during epochs to prevent memorization
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/total}, Acc: {correct/total}')

    # Validation step: assess generalization and decide when to stop training
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    print(f'  Val Loss: {val_loss/val_total}, Val Acc: {val_correct/val_total}')

# ---
# Challenge: Reproducibility and Deployment
# Solution: Evaluate on test set, save model checkpoints with torch.save, and reload as needed
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_loss = test_loss / total
acc = correct / total
print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {acc:.4f}')

# Save model weights for reproducibility/deployment. Later, reload for inference or further training.
torch.save(model.state_dict(), 'covid_classifier.pth')
model.load_state_dict(torch.load('covid_classifier.pth'))
model.eval()
