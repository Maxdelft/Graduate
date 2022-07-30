import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from VAE_IHDA import VariationalAutoencoder, test
import matplotlib.pyplot as plt

''' This function calculates for every model the test and train loss and plots them in different loss function plots. 
This allows to generate loss function plots after the models have been trained.'''

# To DO: 
# Store the 

# Load dataset
dir_models_folder  = '/Volumes/ExtremeSSD/Models/Models_VAE/VAE_lr_1e-07_dim_2_kl_state_False' # directory of folder where the models are stored
dir_subset         = '/Volumes/ExtremeSSD/SHL/2018/extracted_data/subset'                      # directory of training and testing data


test_size      = 0.3    # ratio of testing data 
batch_size     = 100    # Batch size
shuffle        = True   # True := dataloader shuffles data
latent_dim     = 2      # latent dimension of model
n_model        = 264    # Number of model iterations you want to evaluate
device         = 'cpu'  # cpu
kl_state       = False  # kl_state: 'True' := kl loss is included, and z is sampled according to z:= mu + beta*sigma, 
                        # 'False' := kl loss is not included, and and z is equal to mu --> basically training an AE instead of VAE  

## Load x and y
x = np.load(dir_subset + '/acc_gyr_magn_fft_norm_train_0.2.npy')
y = np.load(dir_subset + '/labels_train_0.2.npy')

def create_data_loader(x_train, y_train, x_test, y_test):
    ''' This function creates train and testing dataloader.'''
    trainset        = TensorDataset(torch.tensor(x_train),torch.tensor(y_train))
    testset         = TensorDataset(torch.tensor(x_test),torch.tensor(y_test))
    trainloader     = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    testloader      = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
    return trainloader, testloader

def save_figure(n_model,test_loss,train_loss):
    ''' This function function stores the loss plots in the directory of where the models are stored.'''
    plt.figure(figsize=(10, 7))
    plt.plot(
        test_loss, color='red', linestyle='-', 
        label='validation loss'
        )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(dir_models_folder+ '/test_loss_iter'+str(n_model)+'.png')

    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='red', linestyle='-', 
        label='train loss'
        )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(dir_models_folder+ '/train_loss_iter'+str(n_model)+'.png')

    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
        )
    plt.plot(
        test_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(dir_models_folder+ '/loss_iter'+str(n_model)+'.png')

def main():
    # Load data:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42) 
    # Transform to float32, which is requried if you want to run the model on device 'mps':
    x_train, x_test, y_train, y_test = x_train.astype(np.float32), x_test.astype(np.float32), y_train.astype(np.float32), y_test.astype(np.float32)
    # Create trainloader and testloader:
    trainloader, testloader = create_data_loader(x_train, y_train, x_test, y_test)
    
    # Initialize test_loss_total and train_loss_total:
    test_loss_total  = []
    train_loss_total = []

    # Iterate over n_model and calculate for every model the train and test accuaracy:
    for i in range(1,n_model):
        print('Model:',i)
        model           =  VariationalAutoencoder(latent_dim,kl_state)
        last_model_cp   = torch.load(dir_models_folder + '/model_epoch_'+str(i)+'.pth')
        model.load_state_dict(last_model_cp['model_state_dict'])
        model.to(device)
        test_loss       = test(testloader, model, device, kl_state)
        train_loss      = test(trainloader, model, device, kl_state)
        test_loss_total = np.append(test_loss_total,test_loss)
        train_loss_total = np.append(train_loss_total,train_loss)
    
    # Generate and save loss function plots:
    save_figure(n_model,test_loss_total, train_loss_total) 
    
    # Store  train_loss_total and test_loss_total such that they can be reused and do not have to be reevaluated again
    np.save(dir_models_folder +'/train_loss_total/iter_' + str(n_models) + '.npy' ,train_loss_total)
    np.save(dir_models_folder + '/test_loss_total/iter_' + str(n_models) + '.npy' ,test_loss_total)

main()

