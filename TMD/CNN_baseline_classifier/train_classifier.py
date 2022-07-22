import numpy as np
from torch import softmax
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from torchsummary import summary
from datetime import datetime
import torch.nn.functional as F
from classifier import CNN, train_test_split, train, test, DNN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def calculate_weights(y):
    # This function calculates weights that penalize less occuring classes heavier than frequently occuring classes.
    # In this way all classes are equally treated in the cost function.
    nx, xbins, ptchs = plt.hist(y, bins=[0,1,2,3,4,5,6,7,8]) 
    weights          = nx/float(len(y))           # calculate the fraction of each class in the dataset
    weights          = (1)/weights/sum(1/weights) # calculate weights that are probortional to one over the fraction    
    return torch.Tensor(weights).double()

def remove_zero_class(remove_0_class,x,y):
    # This function romoves zero class if it exists.
    # Furthermore this function scales all labels to the interval [0 and num_labels], 
    # because pytorch requires the lowest class label to be zero.
    #  
    if remove_0_class == True: 
        x = x[(y>0)]
        y = y[(y>0)]-1
    return x,y

def create_data_loader(x_train,x_test,y_train,y_test):
    # Input:
    #        x_train, x_test := training and testing features
    #        y_train, y_test := training and testing labels
    # Output: 
    #        trainloader, testloader  := training and testing dataloader

    trainset        = TensorDataset(torch.tensor(x_train),torch.tensor(y_train))
    testset         = TensorDataset(torch.tensor(x_test),torch.tensor(y_test))
    trainloader     = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    testloader      = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

    return trainloader, testloader
    

class SaveBestModel:
    '''
    Class to save the best model while training. 
    If the current epoch's validation loss is less than the previous validation loss, 
    then save the update the model and save it as best model.
    '''

    def __init__(
        self, best_valid_loss = float('inf')):  
        self.best_valid_loss = best_valid_loss # initialize the self.best_valid_loss with infinity function
    
    def __call__(
        self, current_valid_loss,epoch, model, optimizer, criterion):

        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, '/Volumes/ExtremeSSD/Models/best_model.pth')

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, '/Volumes/ExtremeSSD/Models/final_model.pth')


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('/Volumes/ExtremeSSD/Figures/accuracy.png')
    
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
    plt.savefig('/Volumes/ExtremeSSD/Figures/loss.png')

    


######################################################### Define directories and training hyperparameters ##############################################################################################

# Specify directory of extracted feautes and directory to store the model:
dir_extract_2018 = '/Volumes/ExtremeSSD/SHL/2018/extracted_data'             # directory of extracted features and labels

# Define training hyperparameter:
test_size      = 0.3    # ratio of training data 
batch_size     = 500    # Batch size
epochs         = 10     # Number of epochs
device         = 'cpu'  # Device to run skript on "mps" or "cpu"
remove_0_class = True   # Remove 0 class
shuffle        = True   # During sampling shuffle data or not

# Load features and labels
norm    = False    # True if you want to train on normalized features
dataset = 'train' # decide on which dataset you want to train  --> testset is much smaller!

if norm==True:
    x                = np.load(dir_extract_2018 + '/acc_gyr_magn_fft_'+ dataset + '_norm.npy')      # load extracted training features
    y                = np.load(dir_extract_2018+'/labels_' + dataset +'.npy',allow_pickle=True)     # load extracted training labels
    print('You are training on normalized features :)')
else:
    x                = np.load(dir_extract_2018 + '/acc_gyr_magn_fft_'+dataset+'.npy')      # load extracted training features
    y                = np.load(dir_extract_2018+'/labels_'+dataset+'.npy',allow_pickle=True)     # load extracted training labels



################################################################# main #################################################################################################################################


def main(x,y,remove_0_class,test_size , batch_size, epochs, device, shuffle,model_store_path):

    x,y                              = remove_zero_class(remove_0_class,x,y)                        # Remove the zero class. In the 2018 challenge, there is no zero class. However, toolbox requires the labels to start with 0. 
                                                                                                    # Therefore this function decrease each label by 1 such that lowest label is 0.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42) # Define training data in train and test data
    
    
    # Create dataloader:
    trainloader, testloader = create_data_loader(x_train,x_test,y_train,y_test)

    
    torch.set_default_dtype(torch.float64) # All torch types are of torch.float64, if removed an error appears saying that not all torch tensors are of the same type
    model = CNN()                          # initialize CNN model
    model.to(device)                       # push model to device
    
    weights = calculate_weights(y)

    loss_fn   = nn.CrossEntropyLoss(weight=weights)               # Define Loss Function:
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.00003) # Define optimizer 
    save_best_model = SaveBestModel()

    train_accuracy_total,train_loss_total, test_accuracy_total, test_loss_total = np.array([]), np.array([]),np.array([]),np.array([])

    for epoch in range(epochs):                                       # Start training over the epochs
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_accuracy, train_loss = train(trainloader, model, device, loss_fn, optimizer)
        test_accuracy, test_loss   = test(testloader, model, device, loss_fn)
        print("Done!")
        train_accuracy_total,test_accuracy_total = np.append(train_accuracy_total,train_accuracy), np.append(test_accuracy_total,test_accuracy)
        train_loss_total, test_loss_total        = np.append(train_loss_total,train_loss),np.append(test_loss_total,test_loss)
        save_best_model(
        test_loss, epoch, model, optimizer, loss_fn)
    
    save_plots(train_accuracy_total,test_accuracy_total,train_loss_total,test_loss_total)
    save_model(epochs, model, optimizer, loss_fn)

    


main(x,y,remove_0_class,test_size , batch_size, epochs, device, shuffle)