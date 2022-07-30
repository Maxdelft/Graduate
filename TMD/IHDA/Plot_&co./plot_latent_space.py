import numpy as np
from VAE_IHDA import VariationalAutoencoder, Encoder, AEncoder, Autoencoder
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

''' This function allows to plot the latent space of already trained models.'''

def shuffle_Xy(X,y,N_extract):
    '''This input extraxt N_extract numbers of samples of X and y. Furthermore, X and y are 
       shuffled beforehand.
    '''

    randomize = np.arange(len(y)) # array with index
    np.random.shuffle(randomize)  # shuffle index 
   
    y_shuffle = y[randomize[:N_extract]]   # extract desired number of shuffled labels
    X_shuffle = X[randomize[:N_extract],:] # extract desired number of shuffled features

    return X_shuffle, y_shuffle

def transform_target(y):

    '''This function takes as an input the array containing the labels y and outputs an 
    list with instead of labels(1,2,3, ...8) it contains the labels (Still, Walking, ...., Subway )'''
   
    classes   = ['Still', 'Walking', 'Run', 'Bike','Car','Bus', 'Train','Subway']
    
    y_classes = []
    for y_single in y:
        if y_single == 1:
            y_classes.append(classes[y_single-1])
        elif y_single == 2:
            y_classes.append(classes[y_single-1])
        elif y_single == 3:
            y_classes.append(classes[y_single-1])
        elif y_single == 4:
            y_classes.append(classes[y_single-1])
        elif y_single == 5:
            y_classes.append(classes[y_single-1])
        elif y_single == 6:
            y_classes.append(classes[y_single-1])
        elif y_single == 7:
            y_classes.append(classes[y_single-1])
        elif y_single == 8:
            y_classes.append(classes[y_single-1])

    return y_classes

def plot_dim2(X,y,dir_models,kl_state,n_skip):

    '''
    This function takes as input the directory of the folder where the models are of which the latent space should be plotted.
    It plots the latent space of the desired models. It is requried to specify n_rows and n_cols in this function.
    '''

    ################################################## Specify number of rows and coloumns of subplot here #################################################################################
    n_rows = 4
    n_cols = 5
    ##############################################################################################################################################################################
    classes   = ['Still', 'Walking', 'Run', 'Bike','Car','Bus', 'Train','Subway']
    cdict     = {1:'blue',2:'orange',3:'green',4:'red',5:'brown',6:'pink',7:'cyan',8:'gray'}

    fig, axsrc = plt.subplots(nrows =n_rows, ncols = n_cols)
    count = 1

    for axr in axsrc: 
        for axc in axr:
            torch.set_default_dtype(torch.float64)
            model = VariationalAutoencoder(latent_dims,kl_state)
            model.to(device)

            #Load model parameters:
            last_model_cp = torch.load(dir_models +'model_epoch_'+ str(count)+'.pth' )
            model.load_state_dict(last_model_cp['model_state_dict'])
            model.eval()

            with torch.no_grad():
                Z = model.encoder(X)
            for i in range(1,9):
                ix = np.where(y == i)
                axc.scatter(Z[:,0][ix],Z[:,1][ix], c = cdict[i], label = classes[i-1], s = 10, alpha = 0.5)
                axc.title.set_text('Iter:' + str(count))
                #axc.set_xlim(-2.5,2.5)
                #axc.set_ylim(-2.5,2.5)
            #axc.legend()
            count = count + n_skip
    handles, labels = axc.get_legend_handles_labels()
    fig.legend(handles, labels, loc ='lower right')
    plt.show()

def plot_best(dir_best,X,y,kl_state):

    '''This function plots the latent space of the best performing model'''
    fig, ax = plt.subplots() 
    
    torch.set_default_dtype(torch.float64)
    model = VariationalAutoencoder(latent_dims,kl_state)
    model.to(device)

    #Load model parameters:
    last_model_cp = torch.load(dir_best )
    model.load_state_dict(last_model_cp['model_state_dict'])
    model.eval()

    classes   = ['Still', 'Walking', 'Run', 'Bike','Car','Bus', 'Train','Subway']
    cdict     = {1:'blue',2:'orange',3:'green',4:'red',5:'brown',6:'pink',7:'cyan',8:'gray'}
    
    with torch.no_grad():
        Z = model.encoder(X)
    Z = np.array(Z)

    for i in range(1,9):
        ix = np.where(y == i)
        ax.scatter(Z[:,0][ix], Z[:,1][ix], c = cdict[i], label = classes[i-1], s = 10, alpha = 0.5)
    ax.legend()
    plt.title('Iter: Best')
    plt.show()

def main():
    X    = np.load(dir_X)            # Load X
    y    = np.load(dir_y)            # Load y
    X, y = shuffle_Xy(X,y,N_extract) # Shuffle x and y
    X    = torch.tensor(X)           # Convert X to tensor
    
    plot_dim2(X,y,dir_models,kl_state,n_skip)      # Multiplot
    plot_best(dir_best_model,X,y,kl_state) # Single Plot of only the best iteration


############################################################### Define directories and hyperparameters #############################################################

dir_data       = '/Volumes/ExtremeSSD/SHL/2018/extracted_data/subset/' # directory of considered data
dir_X          = dir_data + 'acc_gyr_magn_fft_norm_train_0.2.npy'      # directory of features
dir_y          = dir_data + 'labels_train_0.2.npy'                     # directory labels 
N_extract      = 8000                                                  # number of latent space samples that should be calculated and plotted
latent_dims    = 2                                                     # latent dimension
device         = 'cpu'                                                 # device on which calculations should be performed
dir_models     = '/Volumes/ExtremeSSD/Models/Models_VAE/VAE_lr_1e-07_dim_2_kl_state_False/' # directory of models
dir_best_model = dir_models +'best_model.pth' # direcotry of best model
kl_state       = False # kl_state := False implies that the kl loss is not included and that the latent space is sampled according to z := mu
n_skip         = 10    # Number of iterations that are skipped in the multiplot


main()