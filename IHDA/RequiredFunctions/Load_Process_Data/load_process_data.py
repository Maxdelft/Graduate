import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt

def normalize_data(x,Q_5,Q_95):
    '''
    Function  : Perform normalization according to baseline classifier paper.
    Operation performed: (x-Q_5)/(Q_95-Q_5)  
    
    Input:
        x          : Featues that need to be normalized
        Q_5 & Q_95 : 5 and 95% quantil
    Output:
         x_norm   : Normalize features
    '''
    
    x_norm = (x-Q_5)/(Q_95-Q_5)
    x_norm = np.minimum(np.maximum(x_norm,0),1)
    
    return x_norm


def load_and_normalize_data(dir_data):

    '''
    Function: This function loads and normalize training and testing data.
    Input:
         dir_data : directory of where the data is stored
    Output:
         x_train, x_test : normalized features
         y_train, y_test : labels rescaled to lowest label equal to zero 
    '''
    # Load training data
    x_train = np.load(dir_data + 'acc_gyr_magn_fft_train.npy')
    y_train = np.load(dir_data + 'labels_train.npy')
    
    # Load test data:
    x_test = np.load(dir_data + 'acc_gyr_magn_fft_test.npy')
    y_test = np.load(dir_data + '/labels_test.npy')
    
    # Remove nan values from training data
    y_train = y_train[~np.isnan(x_train).any(axis=1)]
    x_train = x_train[~np.isnan(x_train).any(axis=1),:]
    
    # Remove nan values form testing data
    y_test = y_test[~np.isnan(x_test).any(axis=1)]
    x_test = x_test[~np.isnan(x_test).any(axis=1),:]
    
    # Calculate 5 and 95 percent Quantile:
    Q_95   = np.quantile(x_train,0.95,axis = 0)
    Q_5    = np.quantile(x_train,0.05,axis = 0)
    
    # Normalize train and test data:
    x_train = normalize_data(x_train ,Q_5,Q_95)
    x_test  = normalize_data(x_test,Q_5,Q_95)

    # Scale lowest label to zero: 
    y_train = y_train - 1
    y_test  = y_test - 1

    # Bring testing data into right tensor format:
    x_train = torch.tensor(x_train.astype(np.float32)).view(-1,753)
    y_train = torch.tensor(y_train.astype(np.float32)).view(-1,)
    
    # Bring training data into right tensor format:
    x_test  = torch.tensor(x_test.astype(np.float32)).view(-1,753)
    y_test  = torch.tensor(y_test.astype(np.float32)).view(-1,) 
    
    return x_train, y_train, x_test, y_test


def generate_dataloader(x,y, batch_size, shuffle):
    '''
    Function: This function generate dataloaders for training and testing

    Input: 
          x : features 
          y : labels
          batch_size      : batch size used when loading batches
    Output:
          dataloader: dataloader
    '''

    # Define train, test and evalset
    dataset   = TensorDataset(x, y)
    
    # Define train, test and evalloader:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

    return dataloader

def train_validate_split(x, y, frac_validate):
    '''
    Function: This function randomly split x into train and validation set, where fraction frac_validate of x
            is used for falidation

    Input: 
        x,y : features and labels
        frac_validation: fraction of x used for validation

    Output:
        x_train, y_train        : features and labels of training data
        x_validatie, y_validate : features and labels of validation data
    '''

    # Shuffle index
    idx = np.random.permutation(len(x))
    
    # Index for training and validation
    idx_train    = idx[:int((1-frac_validate)*len(x))]
    idx_validate = idx[int((1-frac_validate)*len(x)):]
    
    # Training and validation data:
    x_train, y_train = x[idx_train,:], y[idx_train]
    x_validate, y_validate = x[idx_validate,:], y[idx_validate] 
    
    
    return x_train, y_train, x_validate, y_validate


def upsample(x,y):
    '''
    Function: This function is used to upsample underrepresented classes. It is guaranteed that every sample is at least once represented in the upsampled dataset.

    Input:
      x, y : features, labels
    Ouput:
      x_upsample, y_upsample : upsampled features and labels
    '''
    # Number of classes:
    n_classes = int(torch.max(y))
    
    # Initialize lists to store upsampled classes:
    x_upsample, y_upsample = [], []

    # Findet largest represented class
    n_max = 0 
    for i in range(n_classes+1):
        if n_max < len(torch.where(y == i)[0]):
            n_max = len(torch.where(y == i)[0])
    
    # Upsample each class:
    for i in range(n_classes + 1):
        # idx of class i
        idx_i   = torch.where(y == i)[0]
        
        # Extract features and labels of class i:
        x_i,y_i = x[idx_i,:],  y[idx_i]

        # Calculate number of samples that needs to be sampled additionaly
        n_upsample = n_max - len(y_i)
        
        # Random idx of samples used for upsampling:
        idx_upsample = torch.randint(0,len(y_i),(n_upsample,))
        
        # Extract upsampled features and lables:
        x_ext = x_i[idx_upsample,:]
        y_ext = y_i[idx_upsample]
        
        # Append features and labels to x_upsample and y_upsample
        x_upsample.append(x_i)
        x_upsample.append(x_ext)
        y_upsample.append(y_i.view(-1,1))
        y_upsample.append(y_ext.view(-1,1))
        
    x_upsample = torch.vstack(x_upsample)
    y_upsample = torch.vstack(y_upsample)
    
    return x_upsample, y_upsample
