import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from dataLoader import VideoDataset
from model import CNN3D


# paths
all_vids = "../../dissData/allVids"
all_labels = "../../dissData/labels/all_labels.json"
train_vids = "../../dissData/train_vids"
test_vids = "../../dissData/test_vids"
valid_vids = "../../dissData/valid_vids"
train_labels = "../../dissData/train_labels/train_data.json"
test_labels = "../../dissData/test_labels/test_data.json"
valid_labels = "../../dissData/valid_labels/valid_data.json"

def main():
    batch_size = 128
    learning_rate = 0.0001
    epochs = 50

    train_dataset = VideoDataset(train_vids, train_labels, transform=None)
    test_dataset = VideoDataset(test_vids, test_labels, transform=None)
    valid_dataset = VideoDataset(valid_vids, valid_labels, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    classifier = CNN3D(120, 90, 120, 0.2, 256, 256, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    criterion = nn.BCELoss()
    optimiser = optim.AdamW(classifier.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)

            outputs = classifier(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss /len(train_loader)}")

        classifier.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                out = classifier(inputs)
                pred = (out > 0.5).float()

                total += labels.size(0)
                correct += (pred == labels.unsqueeze(1)).sum().item()
            
            accuracy = correct / total
            print(f"validation accuracy is: {accuracy}")

if __name__ == "__main__":
    main()


# Data Loading:

# It defines paths for video data (all_vids), labels (all_labels), and separate paths for training, testing, and validation datasets.
# Creates instances of VideoDataset for each dataset (train_dataset, test_dataset, valid_dataset).
# DataLoader Creation:

# Uses PyTorch's DataLoader to create data loaders for training, testing, and validation datasets (train_loader, test_loader, valid_loader).
# Model Initialization:

# Instantiates a 3D CNN model (classifier) using the CNN3D class.
# Device Configuration:

# Determines whether to use GPU (cuda) or CPU (cpu) and moves the model to the selected device.
# Loss Function and Optimizer:

# Specifies the binary cross-entropy loss (BCELoss) as the loss function.
# Uses AdamW as the optimizer.
# Training Loop:

# Iterates through each epoch.
# Sets the model to training mode (classifier.train()).
# Iterates through batches in the training dataset, computes loss, and performs backpropagation.
# Prints the average loss for the epoch.
# Sets the model to evaluation mode (classifier.eval()).
# Evaluates the model on the validation dataset and prints the validation accuracy.