This folder contains presplitted data for training, testing and validation. Furthermore, the data is already processed, which means 
that nan values are removed and normalization to the data is already performed.

x_train, x_validate , x_test : features of training, validation and testing data
y_train, y_validate , y_test : labels of training, validaion and testing data

x_train_upsample, x_validate_sample : upsample features of training and validation data. In those datasets all classes are 
represented by an equal number of samples. This is achieved by upsampling underrepresented classes in the datasets. Attention, the 
testing data is not upsampled.

y_train_upsample, y_test_upsample  : upsample lables of training and validation data. In those datasets all labels are upsampled.

 
