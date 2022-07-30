import numpy as np
''' This file is used to generate a subset of the considered dataset. The data contained in X_subset and y_subset is shuffled within each trainsportation mode.
However the transportation modes are not shuffled.
'''

def shuffle_Xy(X,y,N_mode):
    ''' This function takes as input the fule dataset (X,y), shuffles them and extracts N_mode samples.'''
    randomize = np.arange(len(y)) # array with index
    np.random.shuffle(randomize)  # shuffle index 
   
    y_shuffle = y[randomize[:N_mode]]   # extract desired number of shuffled labels
    X_shuffle = X[randomize[:N_mode],:] # extract desired number of shuffled features

    return X_shuffle, y_shuffle


def main():
    
    # Initalize list to store X_subset and y_subset:
    X_subset = []
    y_subset = []
    
    # Calcualte how much features of every dataset should be extracted:
    N_mode   = int(1/N_classes*fraction*len(y_full_dataset)) 

    # Start the extract for evyer mode N_mode samples
    for mode in range(1,N_classes + 1):
        y_mode = y_full_dataset[(y_full_dataset == mode)] # extract all labels belonging to the considered class
        X_mode = X_full_dataset[(y_full_dataset == mode)] # extract all features belonging to the considered class
        
        X_mode_shuffle, y_mode_shuffle = shuffle_Xy(X_mode, y_mode, N_mode) # shuffle dataset and extract required subset

        X_subset.append(X_mode_shuffle) # store shuffled subset of features 
        y_subset.append(y_mode_shuffle) # store shuffled subset of labels

    X_subset = np.vstack(X_subset) # transform list to array
    y_subset = np.hstack(y_subset)  # transform list to array

    np.save(dir_save +'/acc_gyr_magn_fft_'+ dataset + '_'+str(fraction)+'.npy',X_subset)
    np.save(dir_save +'/labels_'+ dataset + '_'+str(fraction)+'.npy',y_subset)

    print('Done!')



##################################################################### directoreis and hyperparameter  #############################################

dir_data       = '/Volumes/ExtremeSSD/SHL/2018/extracted_data'    # directory of folder where the full dataset is stored
dataset        = 'train'                                          # train or test dataset
dir_save       = dir_data + '/subset'                             # directory to store extracted dataset
dir_x          = dir_data + '/acc_gyr_magn_fft_'+ dataset +'.npy' # directory of features 
dir_y          = dir_data + '/labels_train.npy'                   # directory of labels
X_full_dataset = np.load(dir_x)                                   # load features
y_full_dataset = np.load(dir_y)                                   # load labels
fraction       = 0.2                                              # fraction of how much of the original dataset should be kept
N_classes      = 8                                                # number of classes/transportation modes



main()
