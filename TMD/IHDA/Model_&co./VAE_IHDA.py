
import torch
from torch import device, nn
import torch.nn.functional as F
import numpy as np


# Split data in train and test
def train_test_split(x,y,train_ratio):
    ''' This function takes as input the features x, labels y and the train_ratio. 
    Accordingly the train ratio is used to evalute split x and y in train and test data. '''
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

class Encoder(nn.Module):
    '''Here the Encoder of the Variational Autoencoder is defined'''
    def __init__(self,  latent_dims,kl_state):
        super(Encoder, self).__init__()
        self.conv1   = nn.Conv1d(  1, 100, kernel_size = 15, stride = 5, padding = 0) 
        self.conv2   = nn.Conv1d(100, 100, kernel_size = 15, stride = 5, padding = 0) 
        self.conv3   = nn.Conv1d(100, 100, kernel_size = 15, stride = 2, padding = 0) 
        self.fc1     = nn.Linear(700, 300)
        self.fc2     = nn.Linear(300, 300)
        self.fc3     = nn.Linear(300, 300)
        self.fc4     = nn.Linear(300,latent_dims) 
        self.fc5     = nn.Linear(300,latent_dims) 
        self.dropout = nn.Dropout(0.25)

        self.N = torch.distributions.Normal(0,1)
        self.kl = 0
        self.kl_state = kl_state


    def forward(self, x):
        x = x.view(-1,1,753)

        # Convolutional layers:
        x = F.leaky_relu((self.conv1(x)))
        x = F.leaky_relu((self.conv2(x)))
        x = F.leaky_relu((self.conv3(x)))
        
        # Flatten layer:
        x = torch.flatten(x,1)
        
        # Fully connected layers
        x = F.leaky_relu((self.fc1(x)))
        x = F.leaky_relu((self.fc2(x)))
        x = F.leaky_relu((self.fc3(x)))
        

        # Latent space:
        mu      = self.fc4(x)
        sigma   = torch.exp(self.fc5(x))
        
        # If kl_state is True you are training a traditional Variational Autoencoder
        if self.kl_state == 'True':
            z       = torch.randn_like(mu).mul(sigma).add_(mu)
            self.kl = torch.sum(sigma**2 + mu**2-torch.log(sigma)-0.5)
        else:
            z = mu
            self.kl = 0

        return z

class Decoder(nn.Module):
    '''Here the Decoder of the Variational Autoencoder is defined.'''
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims,300)
        self.linear2 = nn.Linear(300,300)
        self.linear3 = nn.Linear(300,700)
        self.linear4 = nn.Linear(700,753)

        self.t_conv1 = nn.ConvTranspose1d(100, 100, kernel_size = 16, stride = 2, padding = 0)
        self.t_conv2 = nn.ConvTranspose1d(100,100,  kernel_size = 14, stride = 5, padding = 0)
        self.t_conv3 = nn.ConvTranspose1d(100, 1,   kernel_size = 13, stride = 5, padding = 0)
        self.dropout = nn.Dropout(0.2)

    
    def forward(self, z):

        # Fully connected layers:
        x = F.leaky_relu(self.linear1(z))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear3(x))

        # Reshape to 100 channels:
        x = x.view(-1,100,7)

        # Convolutional layers:
        x = F.leaky_relu(self.t_conv1(x))
        x = F.leaky_relu(self.t_conv2(x))
        x = F.leaky_relu(self.t_conv3(x))
        x = x.view(-1,753)
    
        return x

class ADecoder(nn.Module):
    ''' Here the Decoder of the Autoencoder is defined.'''
    def __init__(self, latent_dims):
        super(ADecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims,300)
        self.linear2 = nn.Linear(300,300)
        self.linear3 = nn.Linear(300,700)
        self.linear4 = nn.Linear(700,753)

        self.t_conv1 = nn.ConvTranspose1d(100, 100, kernel_size = 16, stride = 2, padding = 0)
        self.t_conv2 = nn.ConvTranspose1d(100,100,kernel_size = 14, stride = 5, padding = 0)
        self.t_conv3 = nn.ConvTranspose1d(100, 1,kernel_size = 13, stride = 5, padding = 0)

    
    def forward(self, z):
        # Fully connected layer:
        x = F.relu(self.linear1(z))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        x = x.view(-1,100,7)

        # Convolutional layer:
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = x.view(-1,753)
    
        return x

class AEncoder(nn.Module):
    '''Here the Encoder of the Autoencoder is defined.'''
    def __init__(self,  latent_dims):
        super(AEncoder, self).__init__()
        self.conv1   = nn.Conv1d(  1, 100, kernel_size = 15, stride = 5, padding = 0) 
        self.conv2   = nn.Conv1d(100, 100, kernel_size = 15, stride = 5, padding = 0) 
        self.conv3   = nn.Conv1d(100, 100, kernel_size = 15, stride = 2, padding = 0) 
        self.fc1     = nn.Linear(700, 300)
        self.fc2     = nn.Linear(300, 300)
        self.fc3     = nn.Linear(300, 300)
        self.fc4     = nn.Linear(300,latent_dims) 
        self.dropout = nn.Dropout(0.25)


    def forward(self, x):
        x = x.view(-1,1,753)

        # Convolutional layers:
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        
        # Flatten layer:
        x = torch.flatten(x,1)
        
        # Fully connected layers
        x = F.relu((self.fc1(x)))
        x = F.relu((self.fc2(x)))
        x = F.relu((self.fc3(x)))
        

        # Latent space:
        z      = self.fc4(x)
        
        return z

class Autoencoder(nn.Module):
    ''' Definition of the Autoencoder.'''
    def __init__(self, latent_dims):
        super(Autoencoder,self).__init__()
        self.encoder = AEncoder(latent_dims)
        self.decoder = ADecoder(latent_dims)
    
    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

class VariationalAutoencoder(nn.Module):
    '''Defintion of the Variational Autoencoder.'''
    def __init__(self, latent_dims,kl_state):
        super(VariationalAutoencoder,self).__init__()
        self.encoder = Encoder(latent_dims,kl_state)
        self.decoder = Decoder(latent_dims)
    
    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

def train(dataloader, model, device,optimizer):
    '''Here the trainloop of the Variational Autoencoder is defined.'''
    # Training loop
    # Input: 
    #       dataloader:= training dataloader
    #       model     := deep learning model to optimize
    #       device    := device on which calculations should be perfomed "cpu" or "mps"
    #       loss_fn   := predefined loss function        


    size        = len(dataloader.dataset) # number of training samples in the training dataset
    num_batches = len(dataloader)         # number of batches
    model.train()                         # model.train() tells your model that your are going to train. Reason: There are ayers like dropout, batchnorm
                                          # that behave different on train and test, so the model knows what is going on
    train_running_loss = 0 

    for batch, (X,y) in enumerate(dataloader):  # X,y = dataset, batch keeps track of iteration starts 
                                                # starts at 0 and finishes ones it has iterated through the enitre dataset 
        X = X.to(device)                        # Send x and y to device

        num_sample = X.size()[1] # Number of samples contained in one x (753)

        # Step 1: Compute prediction error
        X_pred = model(X)          # Use data X to predict y by the model
    
        X_pred    = torch.squeeze(X_pred)
        rec_loss  = ((X_pred - X)**2).sum() # Reconstruction loss
        kl_loss   = model.encoder.kl        # kl_divergence

        loss   = rec_loss + kl_loss         # total loss 

        # Step 2: Backpropagation
        optimizer.zero_grad() # Sets all gradients to zero --> requried operation in Pytorch
        loss.backward()       # Computes the gradients of the current tensor
        optimizer.step()      # Performs a single optimization step (parameter update


        train_running_loss += loss.item() 
        if batch % 100 == 0: # Every 100 batches give a status update
            loss, current = loss.item(), batch * len(X)            #  loss.item()  := method that extracts the loss's value as a Python float
                                                                   # batch*len(X)  := current data sample that is processed 
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") # plot the loss and the index of the current data sample

    train_loss      = train_running_loss/(num_batches)
    print(f" Avg train loss: {train_loss:>8f} \n")

    return train_loss

def test(dataloader, model, device):
    ''' Test loop of the Variational Autoencoder.'''
    # Testing loop
    # Input:
    #         dataloader:= dataloader for test data
    #         model     := deep learning model to optimize
    #         device    := device on which calculations should be performed "cpu" or "mps"
    # 
    size        = len(dataloader.dataset) # number of data samples in the testing dataset
    num_batches = len(dataloader)         # number of batches required to go through the whole training dataset
    model.eval()                          # tell the model, that you are evaluating
    test_loss   = 0 

    with torch.no_grad():          # disable gradient calculation
        test_running_loss = 0
        for X, y in dataloader:
            X         = X.to(device)
            X_pred    = model(X)
            X_pred    = torch.squeeze(X_pred)
            rec_loss  = ((X_pred - X)**2).sum()
            kl_loss   = model.encoder.kl
            loss      = rec_loss + kl_loss # Calculate loss function
            test_running_loss += loss.item()
    test_loss = test_running_loss/(num_batches)                    # Average loss over all testing data
    print(f" Avg test loss: {test_loss:>8f} \n")
    return test_loss

def trainAE(dataloader, model, device,optimizer):
    ''' Train loop of the Autoencoder'''

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

    for batch, (X,y) in enumerate(dataloader):  # X,y = dataset, batch keeps track of iteration starts 
                                                # starts at 0 and finishes ones it has iterated through the enitre dataset
        X = X.to(device)                        # Send x and y to device


        # Compute prediction error:
        X_pred = model(X)                # Use data X to predict X_pred by the model
        X_pred = torch.squeeze(X_pred)   # Remove 1 dimension of torch tensor
        loss   = ((X_pred - X)**2).sum() # Calculate loss function

        # Backpropagation:
        optimizer.zero_grad() # Sets all gradients to zero --> requried operation in Pytorch
        loss.backward()       # Computes the gradients of the current tensor
        optimizer.step()      # Performs a single optimization step (parameter update)

        train_running_loss += loss.item() # Update the loss
        if batch % 100 == 0: # Every 100 batches give a status update
            loss, current = loss.item(), batch * len(X)            #  loss.item()  := method that extracts the loss's value as a Python float
                                                                   # batch*len(X)  := current data sample that is processed 
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") # plot the loss and the index of the current data sample

    train_loss      = train_running_loss/(num_batches)
    print(f" Avg training loss: {train_loss:>8f} \n")

    return train_loss

def testAE(dataloader, model, device):
    ''' Test loop of the Autoencoder. '''

    # Testing loop
    # Input:
    #         dataloader:= dataloader for test data
    #         model     := deep learning model to optimize
    #         device    := device on which calculations should be performed "cpu" or "mps"
    # 
    size = len(dataloader.dataset) # number of data samples in the testing dataset
    num_batches = len(dataloader)  # number of batches required to go through the whole training dataset
    model.eval()                   # tell the model, that you are evaluating
 
    test_running_loss = 0
    with torch.no_grad():          # disable gradient calculation     
        for X, y in dataloader:
            X          = X.to(device)
            X          = torch.squeeze(X)    
            X_pred     = model(X)
            loss       = (((X_pred - X)**2).sum()) # Calculate loss function
            test_running_loss += loss.item()       # loss.item() the mean of loss of one minibatch
    test_loss = test_running_loss/(num_batches)    # Average loss over all testing data
    print(f" Avg test loss: {test_loss:>8f} \n")
    return test_loss


