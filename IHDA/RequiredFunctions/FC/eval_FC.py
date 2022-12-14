from sklearn.metrics import f1_score
import sklearn
import torch 
import pandas as pd
import numpy as np

def test_classifier(dataloader, classifier, device,criterion):

    '''
    Function: Used to calculate loss function, accuracy and f1 score of classifier

    Input: 
       dataloader : testdataloader
       classifier : classfifier to evaluate
       device     : device to train on
       criterion  : criterion to test classifier
    Output:
       test_loss  : loss function
       test_f1    : f1 value
       test_acc   : accuracy value
    '''
    classifier.to(device)
    classifier.eval()                        # tell the model, that it is going to be evaluated --> e.g. turns of dropout layers
    
    # Initializes running losses
    test_running_loss, acc_running_loss, f1_running_loss = 0,0,0

    
    with torch.no_grad():
      for i, data in enumerate(dataloader,0):

        # Load data
        X, Y = data                         
        X, Y = X.to(device), Y.to(device)
        Y    = Y.view(-1).to(torch.int64) 

        # Predict X
        Y_pred    = classifier(X)              
        
        # Loss
        loss = criterion(Y_pred,Y)

        # Calculate accuracy:
        acc  = accuracy(Y_pred,Y)

        # Caclulate f1_score:
        f1       = F1_score(Y_pred,Y)

        # Update test_running_loss
        test_running_loss    += loss.item()
        acc_running_loss     += acc.item()
        f1_running_loss      += f1

    # Calculate the test loss
    test_loss       = test_running_loss/(i+1)
    test_acc        = acc_running_loss/(i+1)
    test_f1         = f1_running_loss/(i+1)
    
    return test_loss, test_acc, test_f1


def accuracy(outputs, labels):
    '''
    Function: Calculates accuracy

    Input: 
       Outputs: Softmax value for each class
       labels : True label

    Output:
       Acc  : Accuracy
    '''
    _, preds = torch.max(outputs, dim=1)
    acc      =  torch.tensor(torch.sum(preds == labels).item() / len(preds))
    return acc

def F1_score(outputs,labels):
    '''
    Function to calculate F1 score.

    Input: 
       Outputs: Softmax value for each class
       labels : True label

    Output:
       f1  : F1 score
    '''
    _, preds = torch.max(outputs, dim=1)
    f1 = f1_score(labels.to('cpu'),preds.to('cpu'), average = 'weighted')
    return f1

def classifier_predict(classifier,x,device):
    '''
    This function outputs the predicted labels of classfiier based on features.

    Input:
      classifier : model of classifier
      x          : features
      device     : device to perform calculations on
    Output:
       y_pred    : predicted labels
    '''
    outputs = classifier(x.to(device)).detach().cpu()
    _, y_pred = torch.max(outputs, dim=1)
    return y_pred

def calc_confusion_matrix(y_pred,y):
    '''
    This function calculates the confusion. Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label 
    being i-th class and predicted label being j-th class.

    Input :
      y_preds: predicted labels
      y:     : true lables 
    
    Output:
        conf_matrix: confusion matrix
    '''

    labels = [0, 1, 2, 3,4,5, 6,7]
    conf_matrix = sklearn.metrics.confusion_matrix(y.to(torch.int64), y_pred, labels=labels, sample_weight=None, normalize=None)
    index = ['Still', 'Walking', 'Run', 'Bike','Car','Bus', 'Train','Subway']
    d  = {'Still': conf_matrix[:,0],'Walking':conf_matrix[:,1],'Run':conf_matrix[:,2],'Bike':conf_matrix[:,3],'Car':conf_matrix[:,4],'Bus':conf_matrix[:,5],'Train':conf_matrix[:,6],'Subway':conf_matrix[:,7]}
    conf_matrix = pd.DataFrame(data=d, index = index)
    
    return conf_matrix


def confusion_matrix_to_per(confusion_matrix,n_classes,index):
    confusion_matrix_numpy = confusion_matrix.to_numpy()
    N_class = np.sum(confusion_matrix_numpy,axis = 1)
    N_class = np.full((n_classes,n_classes),N_class).transpose()
    confusion_matrix_per = np.round(np.divide(confusion_matrix_numpy,N_class) * 100, decimals = 2)
    confusion_matrix_per = pd.DataFrame(confusion_matrix_per,columns = index ,index = index)
    return confusion_matrix_per