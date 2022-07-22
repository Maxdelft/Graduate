from re import T
import numpy as np
import torch
from classifier import CNN
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

def test(model, testloader):
    """
    Function to test the model
    """
    # set model to evaluation mode
    model.eval()
    print('Testing')
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            x, y = data
            x    = x.to(device)
            y    = y.to(device)

           
            # forward pass
            outputs = model(x)
            # calculate the accuracy
            _, y_preds = torch.max(outputs.data, 1)

            print('y:',y)
            print('y_preds',y_preds)

            valid_running_correct += (y_preds == y).sum().item()
    # loss and accuracy for the complete epoch
    final_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return final_acc

# Load testing data:

batch_size    = 50
shuffle       = True
device        = 'cpu'
dir_test_data = '/Volumes/ExtremeSSD/SHL/2018/extracted_data'

x_test        = torch.tensor(np.load(dir_test_data + '/acc_gyr_magn_fft_test.npy'))
y_test        = np.load(dir_test_data + '/labels_test.npy') - 1
print('y_test min', min(y_test))
print('y_test max', max(y_test))

# Define testloader:

testset        = TensorDataset(torch.tensor(x_test),torch.tensor(y_test))
testloader     = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)


# Load the model:

torch.set_default_dtype(torch.float64)
model  = CNN()
model.to(device)

last_model_cp = torch.load('/Volumes/ExtremeSSD/Models/best_model.pth')
model.load_state_dict(last_model_cp['model_state_dict'])
model.eval()


test_accuracy = test(model,testloader)
print('Test Accuracy:', test_accuracy)