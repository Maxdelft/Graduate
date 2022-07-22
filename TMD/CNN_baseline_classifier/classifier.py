import numpy as np
from torch import softmax
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from torchsummary import summary
from datetime import datetime
import torch.nn.functional as F


# Split data in train and test
def train_test_split(x,y,train_ratio):
    
    # Input:        
    #               x           := data used as features for training
    #               y           := labels belonging to x
    #               train_ratio := number on the interval [0,1] determining the ratio of how much data should be used for training
    # Output:        
    #               x_train,x_test, y_train, y_test := splitted data into training and testing data

    n = int(train_ratio * len(x))
    x_train = x[0:n,:]
    y_train = y[0:n]

    x_test = x[n+1:,:]
    y_test = y[n+1:]

    return x_train, x_test, y_train, y_test


# Define the model:
class CNN(nn.Module):
    # CNN network:
    # To do      : replace number of features (753) by variable


    def __init__(self):
        super(CNN, self).__init__()
        self.conv1   = nn.Conv1d(1, 100, kernel_size =15, stride = 5, padding = 0) 
        self.conv2   = nn.Conv1d(100, 100, kernel_size = 15, stride = 5, padding = 0) 
        self.conv3   = nn.Conv1d(100, 100, 15, stride = 2, padding = 0) 
        self.batch1  = nn.BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch2  = nn.BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch3  = nn.BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch4  = nn.BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch5  = nn.BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch6  = nn.BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.fc1     = nn.Linear(700, 300)
        self.fc2     = nn.Linear(300, 300)
        self.fc3     = nn.Linear(300, 300)
        self.fc4     = nn.Linear(300,8) 
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = x.view(-1,1,753)
        # Convolutional layer consiting of three convolutional 
        # layers combined with batch normalization and relu activation function
        x = F.relu(self.batch1(self.conv1(x)))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.batch3(self.conv3(x)))
        # x = F.relu((self.conv1(x)))
        # x = F.relu((self.conv2(x)))
        # x = F.relu((self.conv3(x)))
        
        # Flatten layer:
        x = torch.flatten(x,1)
        
        # FC layer consisting of three fully connecected 
        # layer combined with dropout and relu activation function
        x = self.dropout(x)
        x = F.relu((self.fc1(x)))
        x = self.dropout(x)
        x = F.relu((self.fc2(x)))
        x = self.dropout(x)
        x = F.relu((self.fc3(x)))

        # Decision layer:
        x = self.softmax((self.fc4(x)))

        return x


class DNN(nn.Module):
    # CNN network:
    # To do      : replace number of features (753) by variable


    def __init__(self):
        super(DNN, self).__init__()
        self.fc1     = nn.Linear(753, 500)
        self.fc2     = nn.Linear(500, 300)
        self.fc3     = nn.Linear(300, 300)
        self.fc4     = nn.Linear(300, 200)
        self.fc5     = nn.Linear(200,8)
        self.softmax = nn.Softmax(dim = 2)
        
    def forward(self, x):
        x = x.view(-1,1,753)
        #x = torch.flatten(x,1)
        x = torch.relu((self.fc1(x)))
        x = torch.relu((self.fc2(x)))
        x = torch.relu((self.fc3(x)))
        x = torch.relu((self.fc3(x)))
        x = torch.relu((self.fc3(x)))
        x = torch.relu((self.fc4(x)))
        x = self.softmax((self.fc5(x)))
        x = x.view(-1,8)
        return x

def train(dataloader, model, device,loss_fn,optimizer):
    # Training loop
    # Input: 
    #       dataloader:= training dataloader
    #       model     := deep learning model to optimize
    #       device    := device on which calculations should be perfomed "cpu" or "mps"
    #       loss_fn   := predefined loss function        


    size = len(dataloader.dataset) # number of training samples in the training dataset
    num_batches = len(dataloader) 
    model.train()                  # tells your model that your are going to train. Reason: There are ayers like dropout, batchnorm
                                   # that behave different on train and test, so the model knows what is going on
    train_running_loss = 0 
    train_running_correct = 0
    for batch, (X,y) in enumerate(dataloader):  # X,y = dataset, batch keeps track of iteration starts 
                                                # starts at 0 and finishes ones it has iterated through the enitre dataset
        X,y = X.to(device), y.to(device)        # Send x and y to device

        # Step 1: Compute prediction error
        y_pred = model(X)          # Use data X to predict y by the model
        loss   = loss_fn(y_pred,y) # Calculate loss function

        # Step 2: Backpropagation
        optimizer.zero_grad() # Sets all gradients to zero --> requried operation in Pytorch
        loss.backward()       # Computes the gradients of the current tensor
        optimizer.step()      # Performs a single optimization step (parameter update)

        train_running_loss += loss.item() 
        train_running_correct +=  (y_pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 100 == 0: # Every 100 batches give a status update
            loss, current = loss.item(), batch * len(X)            #  loss.item()  := method that extracts the loss's value as a Python float
                                                                   # batch*len(X)  := current data sample that is processed 
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") # plot the loss and the index of the current data sample

    train_loss      = train_running_loss/num_batches
    train_accuraccy = 100*train_running_correct/size


    return train_accuraccy, train_loss

def test(dataloader, model, device, loss_fn):
    # Testing loop
    # Input:
    #         dataloader:= dataloader for test data
    #         model     := deep learning model to optimize
    #         device    := device on which calculations should be performed "cpu" or "mps"
    # 
    size = len(dataloader.dataset) # number of data samples in the testing dataset
    num_batches = len(dataloader)  # number of batches required to go through the whole training dataset
    model.eval()                   # tell the model, that you are evaluating
    test_loss, correct = 0,0,      # 

    with torch.no_grad():          # disable gradient calculation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred,y)
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item() # Number of correct predictions
    test_loss /= num_batches # Average loss over all testing data
    test_accuracy = correct/size*100    # Accuracy
    print(f"Test Error: \n Accuracy: {(test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_accuracy, test_loss.detach().numpy()


