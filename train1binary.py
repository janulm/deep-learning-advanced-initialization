# Test file to to figure out how expensive it is to train 1 binary classifier using Resnet-18
# on cifar 100 dataset (all size 2 subsets of each superclass)
# https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

# Using cifar 100

"""
Training ResNet-18 models for binary classification on pairs of classes from CIFAR-100 using PyTorch is a complex task. We'll break it down into several steps:

Data Preparation: Load CIFAR-100 dataset and create pairs of subclasses for binary classification.
Model Setup: Define and initialize the ResNet-18 model.
Training Loop: Implement the training loop with SGD, learning rate adjustment, and weight decay.
Saving Models: Save the model weights after training each pair.
This script will be quite extensive, and due to the complexity and resource constraints of this environment, I will provide the core parts of the code. I'll also include comments for clarity. The actual training would need to be done on a machine with sufficient computational resources.

Let's start with the data preparation and model setup:
"""


# good resource from Isha: https://docs.ffcv.io/ffcv_examples/cifar10.html



import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import os

# CIFAR-100 Data Preparation
def load_cifar100():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151#file-cifar100_mean_std-py
        # this source details how and what mean and std of the datasets are
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def create_binary_datasets(trainset, testset):
    # CIFAR-100 has 20 superclasses, each containing 5 subclasses.
    # Create a list of pairs of subclasses within the same superclass.
    pairs = [(i, j) for i in range(100) for j in range(i+1, 100) if i // 5 == j // 5]

    binary_datasets = []
    for pair in pairs:
        indices_train = [i for i, x in enumerate(trainset.targets) if x in pair]
        indices_test = [i for i, x in enumerate(testset.targets) if x in pair]

        train_subset = Subset(trainset, indices_train)
        test_subset = Subset(testset, indices_test)

        binary_datasets.append((train_subset, test_subset))
    
    return binary_datasets

# Model Setup
def initialize_model(device):
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2) # Adjusting the final layer for binary classification
    model.to_prop = nn.Softmax(dim=1)
    model = model.to(device)
    return model


# Training Function
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step(running_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

    return model

# Main Function
def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device {device}')


    trainset, testset = load_cifar100()
    print(f'Trainset size: {len(trainset)}')
    binary_datasets = create_binary_datasets(trainset, testset)

    for i, (train_subset, test_subset) in enumerate(binary_datasets):
        model = initialize_model(device)
        train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        model = train_model(model, train_loader, criterion, optimizer, scheduler)

        # Save model weights
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f'models/model_{i}.pth')

if __name__ == '__main__':
    main()


"""
Key Points:
This script assumes that you have PyTorch and torchvision installed in your environment.
CIFAR-100 data is loaded and normalized. The dataset is split into pairs of subclasses.
ResNet-18 is modified for binary classification by adjusting the final fully connected layer.
The training loop includes loss calculation, backpropagation, and learning rate scheduling based on plateau.
Each trained model is saved in a separate file.
Limitations and Additional Steps:
This script does not perform validation or testing. You should add a validation loop to monitor the model's performance on unseen data.
Hyperparameters like learning rate, batch size, and number of epochs are set based on your description. They might need adjustments based on actual training observations
"""