import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
# print('hello world')
#================================================================================================================================================================
#================================================================================================================================================================
#================================================================================================================================================================
data_dir = 'Dataset/GAN/BigGan/train'
# Define transforms for the training and validation sets
data_transforms ={
    "train_transforms": transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(299), 
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),
   "valid_transforms": transforms.Compose([transforms.Resize(300),
                                           transforms.CenterCrop(299),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]), 
    "test_transforms": transforms.Compose([transforms.Resize(300),
                                           transforms.CenterCrop(299),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
}
train_data = 0.8
valid_data = 0.2
# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(data_dir, transform=data_transforms["train_transforms"])#loading dataset
valid_data = datasets.ImageFolder(data_dir, transform=data_transforms["valid_transforms"])
# Obtain training indices that will be used for validation and test
num_train = len(train_data)
indices = list(range(num_train))
# np.random.shuffle(indices)
train_count = int(0.8*num_train)
valid_count = int(0.2*num_train)
test_count = num_train - train_count - valid_count
train_idx = indices[:train_count]
valid_idx = indices[train_count:train_count+valid_count]
# remove_count = int(0.90 * train_count)
# np.random.shuffle(train_idx)
# train_idx = train_idx[:-remove_count]
# remove_count2 = int(0.90 * valid_count)
# np.random.shuffle(valid_idx)
# valid_idx = valid_idx[:-remove_count2]
test_idx = indices[train_count+valid_count:]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32, sampler = valid_sampler)
classes=['ai', 'nature']
use_cuda = torch.cuda.is_available()
# model_transfer.load_state_dict(torch.load('MIXED-test2.pt'))
#===============================================================================================================================================================
#===============================================================================================================================================================
#===============================================================================================================================================================
import torch
from torchvision import models, transforms
from PIL import Image
from torch import nn, optim
classes = ['ai','nature']

model1 = models.resnet50()
for param in model1.parameters():
    param.requires_grad = False
n_inputs = model1.fc.in_features 
last_layer = nn.Linear(n_inputs, len(classes))
model1.fc = last_layer
model1.load_state_dict(torch.load('Model/VQDM_Resnet50.pt'))

model2 = models.resnet50()
for param in model2.parameters():
    param.requires_grad = False
n_inputs = model2.fc.in_features 
last_layer = nn.Linear(n_inputs, len(classes))
model2.fc = last_layer
model2.load_state_dict(torch.load('Model/GLIDE_Resnet50.pt'))

model3 = models.resnet50()
for param in model3.parameters():
    param.requires_grad = False
n_inputs = model3.fc.in_features 
last_layer = nn.Linear(n_inputs, len(classes))
model3.fc = last_layer
model3.load_state_dict(torch.load('Model/GAN_Resnet50.pt'))

model4 = models.resnet50()
for param in model4.parameters():
    param.requires_grad = False
n_inputs = model3.fc.in_features 
last_layer = nn.Linear(n_inputs, len(classes))
model4.fc = last_layer
model4.load_state_dict(torch.load('Model/SDM_Resnet50.pt'))
#===============================================================================================================================================================
#===============================================================================================================================================================
#===============================================================================================================================================================
import torch
import torch.nn as nn
import torch.optim as optim

# Set models to evaluation mode
model1.eval()
model2.eval()
model3.eval()
model4.eval()

# Initialize weights equally
w1, w2, w3, w4 = 0.0, 0.0, 0.0, 0.0

# Define a new class for the ensemble
class DynamicWeightedEnsemble(nn.Module):
    def __init__(self, model1, model2, model3, model4):
        super(DynamicWeightedEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

        # Convert weights to parameters so they can be updated during optimization
        self.weights = nn.Parameter(torch.tensor([w1, w2, w3, w4], requires_grad=True))

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        out4 = self.model4(x)

        # Combine the outputs using dynamic weights with softmax normalization
        weights_softmax = F.softmax(self.weights, dim=0)
        out = (weights_softmax[0] * out1 + weights_softmax[1] * out2 + weights_softmax[2] * out3 + weights_softmax[3] * out4)/4
        return out
        
        # # Combine the outputs using dynamic weights
        # out = self.weights[0] * out1 + self.weights[1] * out2 + self.weights[2] * out3 + self.weights[3] * out4
        # return out

# Create an instance of the dynamic weighted ensemble model
dynamic_ensemble_model = DynamicWeightedEnsemble(model1, model2, model3, model4)
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(dynamic_ensemble_model.parameters(), lr=0.0001, momentum=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dynamic_ensemble_model.to(device)
#===============================================================================================================================================================
#===============================================================================================================================================================
#===============================================================================================================================================================
import sys

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    valid_loss_min = np.inf

    log_file_path = "GATING.txt"  # Choose your desired file path

    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file  # Redirect stdout to the log file

        for epoch in range(1, n_epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0

            model.train()
            for batch_idx, (data, target) in enumerate(loaders['train']):
                if use_cuda:
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            model.eval()
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            print('Epoch: {} \tTraining Loss: {:.5f} \tValidation Loss: {:.5f}'.format(
                epoch,
                train_loss,
                valid_loss))

            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.5f} --> {:.5f}). '.format(
                    valid_loss_min,
                    valid_loss))
                # torch.save(model.state_dict(), 'Gating40%.pt')
                valid_loss_min = valid_loss
            print('Saving Model...')
            torch.save(model.state_dict(), 'Gating10%.pt')

        sys.stdout = sys.__stdout__  # Reset stdout to its original state

    return model

loaders_transfer = {'train': trainloader,
                    'valid': validloader}

model_transfer = train(5, loaders_transfer, dynamic_ensemble_model, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')
