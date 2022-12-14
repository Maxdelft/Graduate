import pandas as pd
import train_VAE
import eval_FC 
import numpy as np

def evaluate_dataset_on_vae(trainloader_pot, trainloader_no_pot, trainloader, device, beta , VAE):  
    '''
    Function: This function generates a pandas dataframe containing the total loss, reconstruction loss and kl loss of a VAE on points with positive potential, no potential and the not splitted dataset.

    Input: 
        trainlaoder_pot, trainloader_no_pot, trainloader: dataloader containing points with postitive potential, negative potential and both combined
        device : device to perform calculations on
        beta   : beta defining VAE
        VAE    : model of Variational Autoencoder
    Output:
        df: dataframe containing stats   
    '''
    train_loss, train_rec_loss, train_kl_loss    = train_VAE.test(trainloader, VAE , device,beta)
    loss_pot, rec_loss_pot, kl_loss_pot          = train_VAE.test(trainloader_pot, VAE , device,beta)
    loss_no_pot, rec_loss_no_pot, kl_loss_no_pot = train_VAE.test(trainloader_no_pot, VAE , device,beta)
    
    d = {'Train': [train_loss, train_rec_loss, train_kl_loss],'Pot': [loss_pot,rec_loss_pot, kl_loss_pot],'No Pot': [loss_no_pot, rec_loss_no_pot, kl_loss_no_pot]}
    df = pd.DataFrame(data=d, index =['Loss', 'Rec Loss', 'KL Loss'] )
    
    return df
    
def evaluate_dataset_on_classifier(trainloader_pot, trainloader_no_pot, trainloader_no_shuffle,classifier,device,criterion):
    '''
    Function: This function generates a pandas dataframe containing the total loss, accuracy and f1 score of a classifier trained on points with positive potential, no potential and the not splitted dataset.

    Input: 
        trainlaoder_pot, trainloader_no_pot, trainloader: dataloader containing points with postitive potential, negative potential and both combined
        device : device to perform calculations on
        beta   : beta defining VAE
        VAE    : model of Variational Autoencoder
    Output:
        df: dataframe containing stats   
    '''

    loss_no_pot, acc_no_pot, f1_no_pot = eval_FC.test_classifier(trainloader_no_pot,classifier, device,criterion)
    loss_pot, acc_pot, f1_pot          = eval_FC.test_classifier(trainloader_pot, classifier, device,criterion)
    train_loss, train_acc, train_f1    = eval_FC.test_classifier(trainloader_no_shuffle, classifier, device,criterion)
    
    d = {'Train': [train_loss, train_acc, train_f1],'Pot': [loss_pot,acc_pot, f1_pot],'No Pot': [loss_no_pot, acc_no_pot, f1_no_pot]}
    df = pd.DataFrame(data=d, index =['Loss', 'Accuracy', 'F1 Score']) 
    
    return df

def stats_pot_dataset( Y_pot, Y_no_pot, y_train):
    '''
    Function: This function prints statistics about the distribution of classes in the dataset with positive potential, negative and both combined.
    
    Y_pot, Y_no_pot, y_train: labels of data with positive potential, negative potential, and combined.

    '''
    # Initialize lists:
    N_classes_pot, N_classes_no_pot, N_classes_train = [], [], []
    per_classes_pot, per_classes_no_pot, per_classes_train = [], [], []

    for i in range(0,8,1):

        n_classes_pot    = np.count_nonzero(Y_pot == i)
        n_classes_no_pot = np.count_nonzero(Y_no_pot == i)
        n_classes_train  = np.count_nonzero(y_train == i)
        
        # Append total numbers of class i
        N_classes_pot.append(n_classes_pot)
        N_classes_no_pot.append(n_classes_no_pot)
        N_classes_train.append(n_classes_train)
        
        # Append fraction of class i
        per_classes_pot.append(round(n_classes_pot/len(Y_pot)*100,1))
        per_classes_no_pot.append(round(n_classes_no_pot/len(Y_no_pot)*100,1))
        per_classes_train.append(round(n_classes_train/(len(y_train))*100,1))
    
    # Append total sum of classes and percentages
    N_classes_pot.append(len(Y_pot))
    N_classes_no_pot.append(len(Y_no_pot))
    N_classes_train.append(len(y_train))
    per_classes_pot.append(100)
    per_classes_no_pot.append(100)
    per_classes_train.append(100)
    
    # Generate dataframe
    index = ['Still', 'Walking', 'Run', 'Bike','Car','Bus', 'Train','Subway','Sum']
    d = {'N X_pot': N_classes_pot, 'N X_no_pot' : N_classes_no_pot, 'N X_train': N_classes_train,'Frac X_pot [%]': per_classes_pot, 'Frac X_no_pot [%]' : per_classes_no_pot, 'Frac X_train [%]': per_classes_train}
    df = pd.DataFrame(data=d, index = index)
    
    return df