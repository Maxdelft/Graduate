import numpy as np
import csv

''' This function alllows to load an existing feature array file and convert it to a .csv file such that those features can be used in google colab.'''

# Directory of the folder where the features are stored
dir_features_folder= '/Volumes/ExtremeSSD/SHL/2018/extracted_data/subset'

# Loade desired features and labels:
acc_gyr_magn_fft_train_02       = np.load(dir_features_folder + '/acc_gyr_magn_fft_train_0.2.npy')
acc_gyr_magn_fft_norm_train_02  = np.load(dir_features_folder + '/acc_gyr_magn_fft_norm_train_0.2.npy')
labels_train_02                 = np.load(dir_features_folder + '/labels_train_0.2.npy')

# Store features and labels as csv file in the same folder, where the original feature files are stored.
np.savetxt(dir_features_folder + '/acc_gyr_magn_fft_train_02.csv', acc_gyr_magn_fft_train_02,delimiter = ',')
np.savetxt(dir_features_folder + '/acc_gyr_magn_fft_norm_train_02.csv', acc_gyr_magn_fft_norm_train_02,delimiter = ',')
np.savetxt(dir_features_folder + '/labels_train_02.csv', labels_train_02 ,delimiter = ',')