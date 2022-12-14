import torch

def initialize_model(model,dir_model):
    ''''
    Function: This function initializes weights of model. 

    Input : 
            model := model with random weights (VAE or classifier)
            dir_model := directory of weights of model
    Output:
           model := model with weights stored at dir model
    '''
    # Load model properties: 
    checkpoint = torch.load(dir_model)
    
    # Initialize weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def store_model(dir_store_folder, model,epochs,optimizer, model_name):
  '''
  Function: 

  Input:
    dir_store_folder : Directory to store model
    model            : Model to store 
    epochs           : Number of Epochs
    optimizer        : Optimizer
    model_name       : Name of the model used for storing
  '''      

  dir_store_model = dir_store_folder + str(model_name)
  torch.save({
              'epoch': epochs+1,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, dir_store_model)