import eval_FC 
import losses 
import torch
import numpy as np

def main_retrain_classifier(trainloader,validationloader,classifier,device, criterion, optimizer,epochs,dir_store_folder):

    '''
    Function: Main function of retraining the classifier.

    Input: 
          trainloader      : dataloader containing training data
          validationloader : dataloader containing validation data
          classifier       : model of classifier
          device           : device to perform calculations on
          criterion        : criterion for training classifier
          optimizer        : optimizer for training classifier
          epochs           : number of retraining epochs
          dir_store_folder : directory of where results of retrained classifier
    '''
    
    # Train and eval loss before retraining:
    train_loss, train_acc, train_f1 = eval_FC.test_classifier(trainloader, classifier, device, criterion)
    valid_loss, valid_acc,valid_f1  = eval_FC.test_classifier(validationloader, classifier, device, criterion)
    
    # Print stats:
    epoch_end(train_loss,train_acc,train_f1,valid_loss, valid_acc,valid_f1,0)
    
    # Store results in list called history:
    result = {'train_loss':train_loss,'train_acc': train_acc,'train_f1': train_f1,'valid_loss': valid_loss,'valid_acc': valid_acc,'valid_f1': valid_f1}
    history = []
    history.append(result)
    
    # Initialize SaveBestModel() class:
    save_best_classifier = SaveBestClassifier()
    save_best_classifier(valid_acc, 0, classifier, optimizer,dir_store_folder)
       

    # Start retraining
    for epoch in range(epochs):
        # Retrain:
        train_loss, classifier, optimizer  = train_classifier(trainloader,classifier,criterion,device,optimizer)
        
        # Calculate training and testing stats: 
        train_loss, train_acc,train_f1    = eval_FC.test_classifier(trainloader, classifier, device, criterion)
        valid_loss, valid_acc, valid_f1   = eval_FC.test_classifier(validationloader, classifier, device, criterion)

        # Store classifier: 
        save_best_classifier(valid_acc, epoch, classifier, optimizer,dir_store_folder)    
    
        # Print stats: 
        epoch_end(train_loss,train_acc,train_f1,valid_loss, valid_acc,valid_f1,epoch+1)
        
        #save_best_classifier(test_acc, epoch, cnn, optimizer,dir_store_folder)
        result = {'train_loss':train_loss,'train_acc': train_acc,'train_f1': train_f1,'valid_loss': valid_loss,'valid_acc': valid_acc,'valid_f1': valid_f1}
        history.append(result)
    
    # Store final classifier:
    store_final_classifier(epoch,classifier,optimizer,dir_store_folder)

    # Plot losses:
    losses.plot_classifier_losses_seperatly(history)

def train_classifier(dataloader,classifier,criterion,device,optimizer):
  '''
  Function: Function to train classifier.
  Input:
        dataloader: dataloader containing training data
        classifier: classifier model
        criterion : criterion for training classifier
        device    : device to perform calculations on
        optimizer : optimizer used for training
    Output:
        train_loss : train loss
        classifier : adjusted classifier
        optimizer  : optimizer
  '''
  # Initialize running losses: 
  running_loss =  0

  for i, data in enumerate(dataloader,0):
    # get the inputs; data is a list of [inputs, labels]
    X, Y = data
    X, Y = X.to(device), Y.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # Predict X
    Y_pred = classifier(X) 
    Y = Y.view(-1).to(torch.int64)

    # Caclulate loss:
    loss = criterion(Y_pred,Y)

    # Backward + Optimize
    loss.backward()
    optimizer.step()
     
    # Update running loss
    running_loss  += loss.item()
   
  train_loss         = running_loss/(i+1)

  return train_loss, classifier, optimizer


def epoch_end(train_loss,train_acc,train_f1,valid_loss, valid_acc,valid_f1,epoch):
    '''
    Function to plot stats about training classifier.

    Input:
         train_loss, valid_loss : loss on training/validation data
         train_acc, valid_acc   : accuracy of training and validation data
         train_f1, valid_f1     : f1 score on train and validation data
    '''
    print("Epoch [{}], train_loss: {:.4f},train_acc: {:.4f},train_f1: {:.4f},valid_loss: {:.4f}, valid_acc: {:.4f}, valid_f1:{:.4f}".format(
        epoch, train_loss, train_acc, train_f1, valid_loss, valid_acc,valid_f1))


class SaveBestClassifier:
    '''
    Description: This function keeps track of which model achieved the best performance on the test set. 
    It automatically saves the new trainined model if it achieves the lowest validation score.

    Output:
    entire_best_model.pth := stores the model, including its structure
    best_model.pth        := stores only the weights
    '''

    def __init__(
        self, best_test_acc = -float('inf')):  
        self.best_test_acc = best_test_acc # initialize the self.best_valid_loss with infinity function
    
    def __call__(
        self, current_test_acc,epoch, classifier, optimizer,dir_store_folder):

        if current_test_acc > self.best_test_acc:
            self.best_test_acc = current_test_acc
            print(f"\nBest validation loss: {self.best_test_acc}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, dir_store_folder + '/best_model.pth')


def store_final_classifier(epoch,classifier,optimizer,dir_store_folder):
    '''
    Function to store final classifier.

    Input: 
        epoch            : number of training epochs
        classifier       : classifer model
        optimizer        : optimizer
        dir_store_folder : direcotry of folder to store final model
    '''
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, dir_store_folder + '/final_model.pth')


def generate_retrainset(X_pot,Y_pot,VAE,device):
    
    '''
    This function calculate for each sample of X_pot a random number N reconstructions of X_pot

    Input: 
       X_pot,Y_pot : features and belonging labels of points with postive potential
       VAE         : model of VAE
       device      : device to perform calculations on

    Output:
       X_pot_retrain, Y_pot_retrain: Array contating features and belonging labels for retraining
    '''
    # Initialize vector containing at each index the number of new samples for that index of X_pot generated 
    #N = np.random.randint(1,10,len(X_pot))
    n   = [3,5,10]
    idx = np.random.randint(0,len(n),len(X_pot))
    N   = [n[i] for i in idx]

    # Initialize list to store X and Y
    X_pot_retrain = [] 
    Y_pot_retrain = []
    

    # Iterate through x_pot
    for idx in range(len(X_pot)):

        # Extrac x_pot and y_pot
        x_pot = torch.ones((N[idx],1)).to(device)*X_pot[idx,:]
        y_pot = torch.ones((N[idx],1)).to(device)*Y_pot[idx]
        # calculate new sample
        x_pot_retrain = calculate_new_sample(VAE,x_pot,device).detach().cpu()
        # Store new samples
        X_pot_retrain.append(x_pot_retrain)
        Y_pot_retrain.append(y_pot)
        
    X_pot_retrain = torch.vstack(X_pot_retrain)
    Y_pot_retrain = torch.vstack(Y_pot_retrain)

    return X_pot_retrain,Y_pot_retrain

def calculate_new_sample(VAE,X,device):
    '''
    Function that generate new sample from X according to z = mu + epsilon*beta*var
    
    Input: 
        VAE:    model of VAE 
        X:      features of sample to create reconstruction from
        device: device to perform calculations on
    
    Output:
        X_rec : reconstructed input 
    '''
    
    Z, mu, logvar = VAE.encoder(X.to(device))
    beta = torch.rand_like(mu)
    z             = beta*torch.randn_like(mu)*torch.exp(0.5*logvar) + mu
    X_rec = VAE.decoder(Z)
    
    return X_rec