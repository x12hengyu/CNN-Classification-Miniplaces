# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # Convolutional layer 1 followed by an ReLU activation layer and a 2D max pooling layer
        # Output shape = 6*28*28
        self.conv2d_1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1,padding=0,bias=True)
        self.relu_1 = nn.ReLU()
        # Output shape = 6*14*14
        self.maxpooling_1 = nn.MaxPool2d(kernel_size=2,padding=0,stride=2)
        # Convolutional layer 2 followed by an ReLU activation layer and a 2D max pooling layer
        # Output shape = 16*10*10
        self.conv2d_2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0,bias=True)
        self.relu_2 = nn.ReLU()
        # Output shape = 16*5*5
        self.maxpooling_2 = nn.MaxPool2d(kernel_size=2,padding=0,stride=2)
        # A Flatten layer to convert the 3D tensor to a 1D tensor.
        self.flatten = nn.Flatten()
        # Fully connected layers
        self.linear1 = nn.Linear(in_features=16*5*5,out_features=256,bias=True)
        self.relu_1_1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=256,out_features=128,bias=True)
        self.relu_1_2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=128,out_features=100,bias=True)
        # certain definitions

    def forward(self, x):
        shape_dict = {}
        # Convolve then relu and maxpooling
        x = self.relu_1(self.conv2d_1(x))
        x = self.maxpooling_1(x)
        shape_dict.update({1: list(x.shape)})
        # Convolve then relu and maxpooling
        x = self.relu_2(self.conv2d_2(x))
        x = self.maxpooling_2(x)
        shape_dict.update({2: list(x.shape)})
        # Flatten
        x = self.flatten(x)
        shape_dict.update({3: list(x.shape)})
        # Fully connected
        x = self.relu_1_1(self.linear1(x))
        shape_dict.update({4: list(x.shape)})
        x = self.relu_1_2(self.linear2(x))
        shape_dict.update({5: list(x.shape)})
        out = self.linear3(x)
        shape_dict.update({6: list(out.shape)})
        # certain operations
        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    for param in model.parameters():
        if param.requires_grad:
            model_params += param.numel()
    
    return model_params/1e6


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
