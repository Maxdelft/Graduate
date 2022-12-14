import numpy
import torch
import os

################ Train test function ######################

def train_test(beta,model,dir_store_folder,optimizer,trainloader,testloader,epochs,device):
  '''
  Function: performs training and testing procedure

  Input:
        beta: beta used for training VAE
        model: model to be optimized
        dir_store_folder: folder where to store best model
        lr              : learning used for training
        trainloader     : dataloader used for training
        testloader      : dataloader used for testing
        epochs          : Number of training epochs
  Output:
        history: list containg training and testing losses
        model: trained model
        optimizer: optimizer
  '''
  
  # Initialize SaveBestModel() class:
  save_best_model = SaveBestModel()

  # Initialize list to store results in:
  history = []
  
  for epoch in range(epochs): # loop over the dataset multiple times
    
    # Train and Test
    train_loss , train_rec_loss , train_kl_loss = train(trainloader, model, device,optimizer,beta)
    test_loss, test_rec_loss, test_kl_loss      = test(testloader, model, device,beta)
     
    # Update save_best_model
    
    save_best_model(test_loss, epoch, model, optimizer,dir_store_folder)
       
    # Store calculated losses in array:
    result = {'train_loss':train_loss,'test_loss': test_loss,'train_rec_loss': train_rec_loss, 'train_kl_loss':train_kl_loss,'test_rec_loss': test_rec_loss, 'test_kl_loss':test_kl_loss}
    history.append(result)
    
    # Print training and testing loss:
    print("Epoch [{}], train_loss: {:.4f},test_loss: {:.4f}".format(
          epoch, train_loss, test_loss))

  return history, model, optimizer

############### Training function #########################

def train(dataloader, model, device,optimizer,beta):

  '''
  Function: Performs training step

  Input:
        datalaoder: dataloader used for training
        model     : model that needs to be optimized
        device    : device to train on
        beta      : beta used for definition of VAE

  Output:
        train_loss     : Total train loss
        train_rec_loss : Reconstruction Loss
        train_kl_loss  : Kullback-Leibler Divergence
  '''
  
  # Initialize running losses: 
  running_loss, rec_running_loss, kl_running_loss  = 0, 0, 0

  model.train()

  for i, data in enumerate(dataloader,0):
    # get the inputs; data is a list of [inputs, labels]
    X, Y = data
    X, Y = X.to(device), Y.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # Predict X
    X_pred = model(X) 

    # Caclulate loss:
    rec_loss = torch.mean(torch.mean((X-X_pred)**2,dim = 1)) 
    kl_loss  = model.encoder.kl

    # Total_loss_function: 
    loss = 500*(753*rec_loss + beta*kl_loss)

    # Backward + Optimize
    loss.backward()
    optimizer.step()
     
    # Update running loss
    running_loss                += loss.item()
    rec_running_loss            += rec_loss.item()
    kl_running_loss             += kl_loss.item()

  
  train_loss         = running_loss/(i+1)
  train_rec_loss     = rec_running_loss/(i+1)
  train_kl_loss      = kl_running_loss/(i+1)

  return train_loss , train_rec_loss , train_kl_loss


############### Testing function ###########################

def test(dataloader, model, device,beta):
  '''
  Function: Performs testing step
  
  Input:
    datalaoder: dataloader used for training
    model     : model that needs to be optimized
    device    : device to train on
    beta      : beta used for definition of VAE

  Output:
    test_loss     : Total test loss
    test_rec_loss : Test reconstruction loss
    test_kl_loss  : Test Kullback Leibler Divergence
  '''
    
  model.eval()                        # tell the model, that it is going to be evaluated --> e.g. turns of dropout layers
    
  # Initializes running losses
  test_running_loss, rec_running_loss, kl_running_loss    = 0, 0, 0  

    
  with torch.no_grad():
    for i, data in enumerate(dataloader,0):

      # Load data
      X, Y = data                         
      X, Y = X.to(device), Y.to(device) 

      # Predict X
      X_pred    = model(X)              

      # Caclulate loss:
      rec_loss = torch.mean(torch.mean((X-X_pred)**2,dim = 1)) 
      kl_loss  = model.encoder.kl

      # Total loss: 
      loss = 500*(753*rec_loss + beta*kl_loss)

      # Update test_running_loss
      test_running_loss    += loss.item()
      rec_running_loss     += rec_loss.item()
      kl_running_loss      += model.encoder.kl.item()

  # Calculate the test loss    
  test_loss       = test_running_loss/(i+1)
  test_rec_loss   = rec_running_loss/(i+1)
  test_kl_loss    = kl_running_loss/(i+1)
    
  return test_loss, test_rec_loss, test_kl_loss

################# Class to save best model ###################

class SaveBestModel:
    '''
    Description: This function keeps track of which model achieved the best performance on the test set. 
    It automatically saves the new trainined model if it achieves the lowest validation score.

    Output:
    entire_best_model.pth := stores the model, including its structure
    best_model.pth        := stores only the weights
    '''
    
    def __init__(
        self, best_valid_loss = float('inf')):  
        self.best_valid_loss = best_valid_loss # initialize the self.best_valid_loss with infinity function
    
    def __call__(
        self, current_valid_loss,epoch, model, optimizer,dir_store_folder):

        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, dir_store_folder + '/best_model.pth')



###################### Function to create directory and store VAE ##################

def create_directory_VAE(beta, latent_dims, epochs,layer,dir_IHDA):
  
  '''
  Function: Creates directory for storing results of training procedure of VAE

  Input: 
    beta        : beta that defines the VAE
    latent_dims : latent dimension of VAE
    epochs      : Epochs of VAE
    layer       : Layer to be optimized
   
   Output:
    dir_store_folder : diectory of where results will be stored
  '''
  
  dir_store_folder = dir_IHDA + 'VAE' +'/VAE_'+str(layer)+'/Models/VAE_beta_'+str(beta) + '_latent_dims_' + str(latent_dims) + '_epochs_' + str(epochs) +'/'
  
  if os.path.isdir(dir_store_folder) == False:
    os.mkdir(dir_store_folder)
    print('Directory is created')
  else:
    print('Directory already exists')
  
  return dir_store_folder