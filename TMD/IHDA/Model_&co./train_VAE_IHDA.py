import numpy as np
from torch import softmax
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from torchsummary import summary
from datetime import datetime
import torch.nn.functional as F
from VAE_IHDA import VariationalAutoencoder,  train, test, trainAE, testAE, Autoencoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os.path import exists
import os

'''This file is used train the Variational Autoencoder or the Autoencoder'''


def create_data_loader(x_train,x_test,y_train,y_test,shuffle):
    ''' 
    Description: This model creates the train and test dataloaders.

    Input:
        x_train, x_test := training and testing features
        y_train, y_test := training and testing labels
     Output: 
       trainloader, testloader  := training and testing dataloader'''

    trainset        = TensorDataset(torch.tensor(x_train),torch.tensor(y_train))
    testset         = TensorDataset(torch.tensor(x_test),torch.tensor(y_test))
    trainloader     = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    testloader      = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

    return trainloader, testloader
    
class SaveBestModel:
    '''
    Description: This function keeps track of which model achieved the best performance on the test set. 
    It automatically saves the new trainined model if it achieves the lowest validation score.

    Output:
    entire_best_model.pth := stores the model, including its structure
    best_model.pth        := stores only the weights
    '''

    def __init__(
        self, best_valid_loss = float('inf')):  
        self.best_valid_loss = best_valid_loss # initialize the self.best_valid_loss with infinity function
    
    def __call__(
        self, current_valid_loss,epoch, model, optimizer,dir_store_folder):

        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, dir_store_folder + '/best_model.pth')
            torch.save(model, dir_store_folder + '/entire_best_model.pth')

def save_model(epoch, model, optimizer,dir_store_folder,kl_state):
    """
    Description: This function saves every new trained model.
    """ 

    print(f"Saving model...")
    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, dir_store_folder +'/model_epoch_'+str(epoch +214+1)+'.pth')

def save_plots(train_loss, valid_loss,dir_store_folder,kl_state):
    """
    Description: This function saves figures of training and validation loss.
    """
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(dir_store_folder + '/loss.png')

    # train loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(dir_store_folder + '/train_loss.png')

    # test loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(dir_store_folder + '/test_loss.png')

def load_model(model_type,dir_store_AE_folder, dir_store_VAE_folder,kl_state):
    ''' 
    Description: This funciton initializes the model. Furthermore, if existing it loads the best_model achieved with the same parameters. If not existing 
    the desired folder is created.
    '''
    if model_type == 'AE':
        print('You are training an Autoencoder')
        model = Autoencoder(latent_dim)
        if exists(dir_store_AE_folder + '/best_model.pth') == True:
            last_model_cp = torch.load(dir_store_AE_folder + '/best_model.pth')
            model.load_state_dict(last_model_cp['model_state_dict'])
        elif exists(dir_store_AE_folder) == False:
            os.mkdir(dir_store_AE_folder)
        dir_store_folder = dir_store_AE_folder
    elif model_type == 'VAE':
        print('You are training an Variational Autoencoder')
        model = VariationalAutoencoder(latent_dim,kl_state)
        if exists(dir_store_VAE_folder + '/best_model.pth') == True:
            last_model_cp = torch.load(dir_store_VAE_folder + '/best_model.pth')
            model.load_state_dict(last_model_cp['model_state_dict'])
            # Perhaps I should first evaluate here before start training --> might automatically update simple because the previous one was not evaluated
        elif exists(dir_store_VAE_folder) == False:
            os.mkdir(dir_store_VAE_folder)
        dir_store_folder = dir_store_VAE_folder
    else:
        print('You need to specify your model')
    return model, dir_store_folder

def train_test(model,trainloader,testloader,device,optimizer):
    '''Calculate train and test loss for desired model.'''
    if model_type == 'AE':
        train_loss = trainAE(trainloader, model, device, optimizer)
        test_loss  = testAE(testloader, model, device)
    elif model_type == 'VAE':
        train_loss = train(trainloader, model, device, optimizer)
        test_loss  = test(testloader, model, device)
    return train_loss, test_loss

######################################################### Define directories and training hyperparameters ##############################################################################################

# Specify directory of extracted feautes and directory to store the model:
dir_extract_2018 = '/Volumes/ExtremeSSD/SHL/2018/extracted_data'             # directory of extracted features and labels
dir_subset       = '/Volumes/ExtremeSSD/SHL/2018/extracted_data/subset'      # directory of subset 
dir_store_VAE    = '/Volumes/ExtremeSSD/Models/Models_VAE'                   # directory of where the VAE folder should be stored
dir_store_AE     = '/Volumes/ExtremeSSD/Models/Models_AE'                    # directory of where the AE_folder should be stored


# Define training hyperparameter:
test_size      = 0.3    # ratio of testing data 
batch_size     = 100    # Batch size
epochs         = 50     # Number of epochs
device         = 'cpu'  # Device to run skript on "mps" or "cpu"
remove_0_class = True   # Remove 0 class
shuffle        = True   # During sampling shuffle data or not
latent_dim     = 2      # Latent dimension of the model
lr             = 1e-07  # learning rate of the model
model_type     = 'VAE'  # 'AE' or 'VAE'
kl_state       = False  # False := only reconstruction loss is considered for training of VAE


## Directory of the folder where the models and figures are stored. If it does not exist it get created.
dir_store_AE_folder  = dir_store_AE  + '/AE_lr_'  + str(lr) + '_dim_' + str(latent_dim) + '_kl_state_' + str(kl_state)
dir_store_VAE_folder = dir_store_VAE + '/VAE_lr_' + str(lr) + '_dim_' + str(latent_dim) +'_kl_state_'  + str(kl_state)

## Load datasubset consisting of features x and labels y
x = np.load(dir_subset + '/acc_gyr_magn_fft_norm_train_0.2.npy')
y = np.load(dir_subset + '/labels_train_0.2.npy')


################################################################# main #################################################################################################################################


def main(x,y,remove_0_class,test_size , batch_size, epochs, device, shuffle,latent_dim):

    # Create train and test loader:
    torch.set_default_dtype(torch.float32) # change dytpe to float32, which is reuqried if the model should be trained on device mps
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42) # Devide training data in train and test data
    x_train, x_test, y_train, y_test = x_train.astype(np.float32), x_test.astype(np.float32), y_train.astype(np.float32), y_test.astype(np.float32) # convert train and test dataset to float32 which is required to be able to run the optimization on the device "mps"

    # Create dataloader:
    trainloader, testloader = create_data_loader(x_train,x_test,y_train,y_test,shuffle)     

    # Load model:
    model, dir_store_folder = load_model(model_type,dir_store_AE_folder, dir_store_VAE_folder,kl_state)
    model.to(device)                         
    
    # Define optimizer:
    optimizer       = torch.optim.Adam(model.parameters(),lr = lr, eps=1e-40) 
    save_best_model = SaveBestModel()

    # Initialize array:
    train_loss_total, test_loss_total = np.array([]), np.array([])
    
    # Start training and iterate over epochs:
    for epoch in range(epochs):  
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss, test_loss             = train_test(model, trainloader,testloader, device, optimizer)
        print("Done!")
        train_loss_total, test_loss_total = np.append(train_loss_total,train_loss), np.append(test_loss_total,test_loss)
        save_best_model(test_loss, epoch, model, optimizer,dir_store_folder)
        save_model(epoch, model, optimizer,dir_store_folder,kl_state)

    save_plots(train_loss_total,test_loss_total,dir_store_folder,kl_state)


main(x,y,remove_0_class,test_size , batch_size, epochs, device, shuffle,latent_dim)