import numpy as np


def extract_sensor_data(acc_gyr_magn_fft):
    '''This function extracts acc, gyr and magn data of the combined matrix acc_gyr_magn_fft'''


    x_n = int(np.shape(acc_gyr_magn_fft)[1]/3) # Number of data samples belonging to one sensor
    
    acc_fft = acc_gyr_magn_fft[:,:x_n]
    gyr_fft = acc_gyr_magn_fft[:,x_n:2*x_n]
    magn_fft = acc_gyr_magn_fft[:,2*x_n:]

    return acc_fft, gyr_fft, magn_fft

def get_percentile(x):
    ''' This function calculate the 5 (x_5), and 95 (x_95) percentile of the input matrix x'''
    x_95 = np.nanpercentile(x, 95)
    x_5  = np.nanpercentile(x,  5)
    return x_5, x_95

def normalize(x,x_5,x_95):

    '''This function normalizes the input data x, by incorporating the 5 (x_5) and the 95 (x_95) percentile'''

    #x_norm = np.minimum(np.maximum((x-x_5)/(x_95-x_5),0),1)
    x_norm  = np.maximum((x-x_5)/(x_95-x_5),0)
    
    return x_norm




###################################################################################### Specify directory where data set that is going to be normalized is stored  #############################################################################


dir_extract    = '/Volumes/ExtremeSSD/SHL/2018/extracted_data/' # directory where acc_gyr_magn is stored and where normalized data set will be stored

def main():
    ''' 
        This function normalizes the previously calculate features named acc_gyr_magn_fft, calculated in the file evaluate_classifier.
     '''

    # Load data sets that are going to be normalized
    acc_gyr_magn_fft_train  = np.load(dir_extract + 'acc_gyr_magn_fft_train.npy')
    acc_gyr_magn_fft_test   = np.load(dir_extract + 'acc_gyr_magn_fft_test.npy')

    # Extract acceleromete gyroscope and magnetmeter measurments of the data set
    acc_fft_train, gyr_fft_train, magn_fft_train = extract_sensor_data(acc_gyr_magn_fft_train)
    acc_fft_test , gyr_fft_test , magn_fft_test  = extract_sensor_data(acc_gyr_magn_fft_test)

    print('Train:',gyr_fft_train[0:10,0:10])
    print('Test:' ,gyr_fft_test[0:10,0:10])

    # Cacluate for accelerometer, gyroscope and magnetometer the 5 and 95 percent quantiles
    acc_fft_5, acc_fft_95   = get_percentile(acc_fft_train)
    gyr_fft_5, gyr_fft_95   = get_percentile(gyr_fft_train)
    magn_fft_5, magn_fft_95 = get_percentile(magn_fft_train)
    print('Acc_fft_95:', acc_fft_95)
    print('Acc_fft_5 :', acc_fft_5)

    # Normalized training and testing data of accelerometer, gyroscope and magnetometer by using the 5 and 95 percent quantiles
    acc_fft_norm_train  = normalize(acc_fft_train, acc_fft_5, acc_fft_95)
    gyr_fft_norm_train  = normalize(gyr_fft_train, gyr_fft_5, gyr_fft_95)
    magn_fft_norm_train = normalize(magn_fft_train, magn_fft_5, magn_fft_95)

    acc_fft_norm_test  = normalize(acc_fft_test,acc_fft_5,acc_fft_95)
    gyr_fft_norm_test  = normalize(gyr_fft_test,gyr_fft_5,gyr_fft_95)
    magn_fft_norm_test = normalize(magn_fft_test,magn_fft_5,magn_fft_95)

    print('Test  : ',acc_fft_norm_test[0:10,0:10])
    print('Train : ',acc_fft_norm_train[0:10,0:10])

    # Stack normalized accelerometer, gyroscope and magnetometer data together to one array 
    acc_gyr_magn_fft_norm_train = np.hstack((acc_fft_norm_train,gyr_fft_norm_train,magn_fft_norm_train))
    acc_gyr_magn_fft_norm_test  = np.hstack((acc_fft_norm_test,gyr_fft_norm_test,magn_fft_norm_test))
  
    # Save normalized train and test data set
    np.save(dir_extract + 'acc_gyr_magn_fft_norm_train.npy', acc_gyr_magn_fft_norm_train)
    np.save(dir_extract + 'acc_gyr_magn_fft_norm_test.npy' , acc_gyr_magn_fft_norm_test)

    print('Data is normalized and stored!')


main()





