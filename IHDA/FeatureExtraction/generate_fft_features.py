import numpy as np
from scipy.fft import fft, fftfreq

def calc_mag(dir_shl_imu):
    '''
    Description: This function takes as an input the directory of the imu file. The imu file contains seperate files for measurements of 
                  accelerometer, gyroscope and magnetometer for x,y and z direction.
    Input: 
            dir_shl_imu :  directory of imu file containing imu sensor data (train or test)
    Output:
            sensor_mag  : list of three arrays containing magnitude values of accelerometer, gyroscope and magnetometer
    '''
    imu =      ['Acc_','Gyr_','Mag_']       # names of imu files 
    imu_axis = ['x.txt','y.txt','z.txt']    # imu axis

    # Initialize list to store magnitude of sensors in order accelerometer, gyroscope and magnetometer:
    sensor_mag = []          
    for sensor in imu:        # Loop iterating over different imu sensors
        sensor_xyz = []       # List to store x,y,z data of one sensor
        for axis in imu_axis: # Loop iterating over different axis of imu sensor
            data = np.loadtxt(dir_shl_imu + sensor + axis) # load imu file
            sensor_xyz.append(data)
        sensor_magni = np.sqrt(np.square(sensor_xyz[0])+np.square(sensor_xyz[1])+np.square(sensor_xyz[2]))
        sensor_mag.append(sensor_magni)
    return sensor_mag

def calc_fft_feat(sensor_mag, N_meas, N_skip): 
    ''' 
    Description: This function takes as an input the magnitude values of accelerometer, gyroscope and magnetometer 
                  and outputs an array containing the magnitude frequency values of those three sensors
    Input:
        sensor_mag : list containg three arrays with magnitude data of accelerometer, gyroscope and magnetometer 
                         coloumns contain 60000 measurments resulting from sampling rate of 100Hz for 60seconds
        N_meas : number of considered measurements 
        N_skip : number of skipped measurements
                        
    Output: 
        acc_gyr_magn_fft : each row represents one sample containing fft values of accelerometer, gyroscope and magnetometer
    '''
    mag_acc  = sensor_mag[0] # Extract magnitude values for accelerometer
    mag_gyro = sensor_mag[1] # Extract magnitude values for gyro
    mag_magn = sensor_mag[2] # Extract magnitude values for magnetometer 
 

    t_start = 0  # Initilaize t_start
    acc_gyr_magn_fft = np.zeros((3*int((N_meas)/2+1))) # Initialize array to store combined values in
    
    while t_start <= len(mag_acc[2])-N_meas:
        t_end    = t_start + N_meas
        fft_acc  = np.abs(fft(mag_acc[:,t_start:t_end],axis = 1)[:,0:int((N_meas)/2+1)])  # Calculate magnitude fourier values for accelerometer
        fft_gyro = np.abs(fft(mag_gyro[:,t_start:t_end],axis = 1)[:,0:int((N_meas)/2+1)]) # Calculate magnitude fourier values for gyro
        fft_magn = np.abs(fft(mag_magn[:,t_start:t_end],axis = 1)[:,0:int((N_meas)/2+1)]) # Calculate magnitude fourier values for magnetometer

        fft_features      = np.hstack((fft_acc,fft_gyro,fft_magn))      # Combine fft_acc, fft_gyro, fft_magn to one array
        acc_gyr_magn_fft  = np.vstack([acc_gyr_magn_fft ,fft_features]) # Store combined array in total array
    
        t_start  += N_skip
    
    acc_gyr_magn_fft      = acc_gyr_magn_fft[1:,:] # Remove first row of zeros

    return acc_gyr_magn_fft

def extract_label(labels, N_meas,N_skip): 
    ''' 
    Input: 
        labels := array with labels, where each single measurement of the considered time frame is labeled
        N_meas := number of considered measurements 
        N_skip := number of skipped measurements
    Output:
        y      := one lable for each sample of measurments
    '''

    t_start = 0                           # Initialize t_start
    y       = []                          # Initialize list to store extracted labels
    labels  = labels.astype(int).tolist() # Convert labels from array to list

    while t_start <= np.shape(labels)[1]-N_meas:
        t_end = t_start + N_meas     # Calculated t_end
        for i in range(len(labels)): # Iterate through rows
            labels_t  = labels[i][t_start:t_end]               # Extract label of i-th row and considered time sequence
            y.append(max(set(labels_t), key = labels_t.count)) # Store most frequent occuring label in final label list
        t_start += N_skip                                      # Update t_start
    return np.array(y)                                         # Output extracted labels as array



################################################## Main ###############################################################################     
def main(dir_shl_data, dir_features):
    # Desciption: This function reads the sensor measurment values and calculates the fourier magnitudes values. Furthermore the labels belonging to one measurment sample
    #             are also calcualted 
    #             One row of acc_gyr_mag contains the fourier magnitude values of one conisdered timeframe. The belonging label can be found in the labels file.
    dir_shl_data_train = dir_shl_data + 'train/'    # train directory containing all training files
    dir_shl_data_test  = dir_shl_data + 'test/'      # test  directory containing all testing files

    dir_shl_train_imu   =  dir_shl_data_train + 'imu/' # directory of file containing train imu data
    dir_shl_train_label =  dir_shl_data_train + 'Label.txt' # directory of file containing train label
    dir_shl_test_imu    =  dir_shl_data_test  + 'imu/'  # directory of file containing test imu data
    dir_shl_test_label  =  dir_shl_data_test  + 'Label.txt' # directory of file containing test label
   
    # Define N_skip and N_meas
    N_skip_train = 250 # Sliding window moves with 2.5s/250data samples
    N_skip_test  = 500
    N_meas       = 500 # Take time frame of 5 seconds
    
    label_train      = np.loadtxt(dir_shl_train_label)
    label_test       = np.loadtxt(dir_shl_test_label)

    sensor_mag_train = calc_mag(dir_shl_train_imu)
    sensor_mag_test  = calc_mag(dir_shl_test_imu)

    acc_gyr_magn_fft_train = calc_fft_feat(sensor_mag_train,N_meas,N_skip_train)
    acc_gyr_magn_fft_test  = calc_fft_feat(sensor_mag_test,N_meas,N_skip_test)
    
    labels_train           = extract_label(label_train,N_meas, N_skip_train)
    labels_test            = extract_label(label_test ,N_meas, N_skip_test)

    np.save(dir_features+'labels_train.npy', labels_train)
    np.save(dir_features +'labels_test.npy', labels_test)
    np.save(dir_features +'acc_gyr_magn_fft_train.npy', acc_gyr_magn_fft_train)
    np.save(dir_features +'acc_gyr_magn_fft_test.npy',acc_gyr_magn_fft_test)




dir_shl_data  = '/Users/maximilianvanamerongen/Desktop/IHDA/SHLData/2018/' # Directory of stored data
dir_features  = dir_shl_data + 'Features/' # Directory where extracted features get stored

main(dir_shl_data, dir_features)