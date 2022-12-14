# Calculate for every point its potential
import torch
import os
import potential 
import load_process_data as lpd
import FC 
import retrain_FC
import basic_functions
import torch.optim as optim
from torch import device, nn
import eval_FC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main_hyperparamter_w(ws,x_train,y_train,x_validate,y_validate,x_test,y_test,Z_train, gamma,dist_metric, device,dir_hyper_folder,dir_fc_base,epochs,dir_VAE,VAE,batch_size):

  '''
  Input:
    ws                     : List containing ws that should be considerd for hyperparametersearch
    x_train, y_train       : x_train, y_train features and labels of training data
    x_validate, y_validate : x_validate, y_validate features and labels of validation data
    x_test, y_test         : x_test, y_test features and labels of test data
    Z_train                : latent space representation of training data
    gamma                  : hyperparameter for potential step calculation
    dist_metric            : distance metric used for calculating distance between points
    dir_hyper_folder       : directory of folder to store results in
    dir_fc_base            : directory of folder containing baseline classifier
    epochs                 : number of retraining epochs
    dir_VAE                : directory of VAE
    VAE                    : Model of Variational Auotencoder
    batch_size             : batch size used for generating dataloaders
  '''
  
  for w in ws:
    ########################## Find points with positive potential ###############################
    potential_ = potential.main_potential(Z_train,y_train,w,gamma,dist_metric,device)

    # Index of points with positive potential of the training data:
    idx_x_train_pot     = torch.where(potential_ > 0)[0] 
  
    # Generate folder to store results:
    store_folder = dir_hyper_folder + 'w_'+str(w)
    
    if os.path.isdir(store_folder) == False:
      os.mkdir(store_folder)#
      
    # Store results in dictonary
    results_w = {'idx_x_train_pot' : idx_x_train_pot,
                 'potential'       : potential_,
                 'model_state_dict': VAE.state_dict(), # Store VAE used
                 'VAE_1_directory' : dir_VAE,
                 }
                 
    # Store results on drive:
    torch.save(results_w,store_folder + '/results_w_' + str(w) + '.pth') 
    
    # If there are no points with positive potential, do not proceed!
    if (results_w['idx_x_train_pot'].numel()== 0) is True:
      print('No points with positive potential!')
      break
      

    # Empty cuda cache
    torch.cuda.empty_cache()
    
    ################################# Retrain classifier ######################################################
    # Split in points with positive and negative potential:

    X_pot, Y_pot, _, _, _, _= potential.split_potential(x_train,y_train,Z_train,potential_,device)
    
    # Generate dataloaders:
    trainloader_pot        = lpd.generate_dataloader(X_pot,Y_pot, batch_size, shuffle = False)
    validationloader       = lpd.generate_dataloader(x_validate,y_validate, batch_size, shuffle = False)
    testloader             = lpd.generate_dataloader(x_test,y_test, batch_size, shuffle = False)

    # Generate dataloader of reconstructed signal:
    X_pot_rec_retrain, Y_pot_rec_retrain = retrain_FC.generate_retrainset(X_pot,Y_pot,VAE,device)

    trainloader_pot_rec                  = lpd.generate_dataloader(torch.tensor(X_pot_rec_retrain),torch.tensor(Y_pot_rec_retrain), batch_size, shuffle = False)

    del X_pot, Y_pot, X_pot_rec_retrain, Y_pot_rec_retrain

    # Define trainloaders on which retraining should be performed:
    trainloaders      = [ trainloader_pot_rec,trainloader_pot]
    trainloaders_name = ['trainloader_pot_rec','trainloader_pot']
    
    results = []
    for i ,dataloader_training in enumerate(trainloaders):
      # Initialize retrain classifier with desired weights
      dropout_fc    = 0.25
      fc_retrain_1  = FC.FC(dropout_fc).to(device)
      fc_retrain_1  = basic_functions.initialize_model(fc_retrain_1,dir_fc_base)
      
      # Initialize optimizer:
      optimizer  = optim.Adam(fc_retrain_1.parameters(), lr =  0.00001,betas=(0.9, 0.999)) 
      
      # Generate folder to store results of current iteration:
      dir_store_retrain = store_folder +'/'+ trainloaders_name[i]

      if os.path.isdir(dir_store_retrain) == False:
        os.mkdir(dir_store_retrain)
      
      # Retrain classifier:
      criterion = nn.CrossEntropyLoss()
      retrain_FC.main_retrain_classifier(dataloader_training,validationloader ,fc_retrain_1,device, criterion,optimizer,epochs,dir_store_retrain)
      
      # Load best classifier:
      # Hyperparamter of retrain classifier:
      dir_fc_best        = dir_store_retrain + '/best_model.pth' 
      
      # Initialize retrain classifier with desired weights
      fc_best  = FC.FC(dropout_fc).to(device)
      fc_best  = basic_functions.initialize_model(fc_best,dir_fc_best)
      
      train_loss, train_acc,train_f1 = eval_FC.test_classifier(dataloader_training, fc_best, device, criterion)
      valid_loss, valid_acc,valid_f1 = eval_FC.test_classifier(validationloader, fc_best, device, criterion)
      test_loss, test_acc,test_f1    = eval_FC.test_classifier(testloader, fc_best, device, criterion)
      
      # Store results of best classifier:
      result = {'train_loss' : train_loss,
                'train_acc'  : train_acc,
                'train_f1'   : train_f1,
                'valid_loss' : valid_loss,
                'valid_acc'  : valid_acc,
                'valid_f1'   : valid_f1, 
                'test_loss'  : test_loss,
                'test_acc'   : test_acc,
                'test_f1'    : test_f1,
                'trainloader': trainloaders_name[i],
                }
      results.append(result)
    dir_result = store_folder + '/retrain_results.pth'
    torch.save(results,dir_result)

  torch.cuda.empty_cache()


def split_results(dir_hyper_folder, ws):

  '''
  Function: This function is used for extracting seperate losses of the hyperparametersearch. 

  Input: 
      dir_hyper_folder : folder of results hyperparameter search
      ws               : list containing considered ws for hyperpaprameter search
  
  Output:
     test_acc_pot, valid_acc_pot           : list containing test and validation accuracy
     test_loss_pot, valid_loss_pot         : list containing test and validation loss
     test_acc_pot_rec, valid_acc_pot_rec   : list containing test and validation accuracy for reconstructed dataset
     test_loss_pot_rec, valid_loss_pot_rec : list containing test and validation loss for reconstructed dataset
     ws                                    : list containing ws
  '''


  results_pot_rec = []
  results_pot     = []
  
  for w in ws:
    dir_results_w = dir_hyper_folder + 'w_'+str(w)
    results_w = torch.load(dir_results_w+'/retrain_results.pth')
    
    valid_loss_pot_rec = results_w[0]['valid_loss']
    valid_acc_pot_rec = results_w[0]['valid_acc']
    
    valid_loss_pot = results_w[1]['valid_loss']
    valid_acc_pot = results_w[1]['valid_acc']
    
    test_loss_pot_rec = results_w[0]['test_loss']
    test_acc_pot_rec = results_w[0]['test_acc']
    
    test_loss_pot = results_w[1]['test_loss']
    test_acc_pot = results_w[1]['test_acc']
    
    result_pot_rec = {'valid_loss':valid_loss_pot_rec,
                   'valid_acc': valid_acc_pot_rec,
                   'test_loss':test_loss_pot_rec,
                   'test_acc':test_acc_pot_rec,
                    'w':w,
                   }
    result_pot     = {'valid_loss':valid_loss_pot,
                   'valid_acc': valid_acc_pot,
                   'test_loss':test_loss_pot,
                   'test_acc':test_acc_pot,
                    'w': w,
    }
    results_pot_rec.append(result_pot_rec)
    results_pot.append(result_pot)

  test_acc_pot = [result['test_acc'] for result in results_pot] 
  valid_acc_pot = [result['valid_acc'] for result in results_pot] 
    
  test_loss_pot = [result['test_loss'] for result in results_pot] 
  valid_loss_pot = [result['valid_loss'] for result in results_pot] 
    
  test_acc_pot_rec = [result['test_acc'] for result in results_pot_rec] 
  valid_acc_pot_rec = [result['valid_acc'] for result in results_pot_rec] 
    
  test_loss_pot_rec = [result['test_loss'] for result in results_pot_rec] 
  valid_loss_pot_rec = [result['valid_loss'] for result in results_pot_rec] 
    
  ws = [result['w'] for result in results_pot_rec] 

  return test_acc_pot, valid_acc_pot, test_loss_pot, valid_loss_pot, test_acc_pot_rec, valid_acc_pot_rec, test_loss_pot_rec, valid_loss_pot_rec, ws

def plot_hyperparameter_losses(test_acc, valid_acc, test_loss, valid_loss, ws,title):
  '''
  Function: This function plots train, test and evaluation loss of classifier aswell as evolution of accurcy for retrained classifier using values contained in w

  Input:
       test_acc, valid_acc   : test and validation accuracy for different ws
       test_loss, valid_loss : test and validation loss for different ws
  '''

  idx_valid_acc_max = valid_acc.index(max(valid_acc)) 
  # Initialize plot:
  fig, axs  = plt.subplots(1,2)
  fig.suptitle(title)
    
  # Plot losses:
  axs[0].plot(ws,valid_loss,label = 'valid')
  axs[0].plot(ws,test_loss,label  = 'test')
  axs[0].plot(ws[idx_valid_acc_max],valid_loss[idx_valid_acc_max],'*',markersize = 12)
  axs[0].plot(ws[idx_valid_acc_max],test_loss[idx_valid_acc_max],'*',markersize = 12)
  axs[0].legend()
  axs[0].set_xlabel('Epochs')
  axs[0].set_ylabel('Loss')

  
  # Plot accuracies:
  axs[1].plot(ws,valid_acc,label = 'valid')
  axs[1].plot(ws,test_acc,label = 'test')
  axs[1].plot(ws[idx_valid_acc_max],valid_acc[idx_valid_acc_max],'*',markersize = 12)
  axs[1].plot(ws[idx_valid_acc_max],test_acc[idx_valid_acc_max],'*',markersize = 12)
  axs[1].legend()
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel('Accuracy')
  plt.show()
  plt.close()


def print_stats(valid_acc_pot,test_acc_pot,valid_acc_pot_rec,test_acc_pot_rec,test_acc_base,valid_acc_base,ws,dir_hyper_folder):
  
  '''
  This function prints some stats. 

  Input:
    valid_acc_pot,test_acc_pot         : validation and test accuracy of classifier retrained on samples with positive potential
    valid_acc_pot_rec,test_acc_pot_rec : validation and test accuracy of classifier retrained on reconstructed samples with positive potential
    test_acc_base,valid_acc_base       : test and validation accuracy of baseline classifier
    ws                                 : tested ws
    dir_hyper_folder                   : directory of hyperparameter folder
  '''
  
  idx_pot_max_valid_acc = valid_acc_pot.index(max(valid_acc_pot))
  
  w_pot_max            = ws[idx_pot_max_valid_acc]
  valid_acc_pot_max    = valid_acc_pot[idx_pot_max_valid_acc]
  test_acc_pot_opt     = test_acc_pot[idx_pot_max_valid_acc]
  test_acc_pot_max     = max(test_acc_pot)
  test_acc_pot_improvement = test_acc_pot_opt - test_acc_base
  valid_acc_pot_improvement = valid_acc_pot_max - valid_acc_base
    
  dir_results_pot_max = dir_hyper_folder + 'w_'+str(w_pot_max) +'/results_w_'+str(w_pot_max) +'.pth'
  results_w_pot_max   = torch.load(dir_results_pot_max) 
  N_pot_max           = len(results_w_pot_max['idx_x_train_pot']) 
  idx_pot_rec_max_valid_acc = valid_acc_pot_rec.index(max(valid_acc_pot_rec))
  w_pot_rec_max             = ws[idx_pot_rec_max_valid_acc]
  valid_acc_pot_rec_max     = valid_acc_pot_rec[idx_pot_rec_max_valid_acc]
  valid_acc_pot_rec_improvement = valid_acc_pot_rec_max - valid_acc_base
    
  dir_results_pot_rec_max = dir_hyper_folder + 'w_'+str(w_pot_rec_max) +'/results_w_'+str(w_pot_rec_max) +'.pth'
  results_w_pot_rec_max   = torch.load(dir_results_pot_rec_max) 
  N_pot_rec_max           = len(results_w_pot_rec_max['idx_x_train_pot']) 
    
  test_acc_pot_rec_opt      = test_acc_pot_rec[idx_pot_rec_max_valid_acc]
  test_acc_pot_rec_max      = max(test_acc_pot_rec)
  test_acc_pot_rec_improvement = test_acc_pot_rec_opt - test_acc_base
    
  index = ['Retrain on potential points', 'Retrain on reconstructions of potential points']
  
  d = {'Optimal w': [w_pot_max, w_pot_rec_max],
       'Validation Accuracy base': [valid_acc_base, valid_acc_base],
       'Optimal validation accuracy':[valid_acc_pot_max, valid_acc_pot_rec_max],
       'Valid Accuracy Improvement':[valid_acc_pot_improvement,valid_acc_pot_rec_improvement],
       'Test Accuracy Base':[test_acc_base,test_acc_base],
       'Test accuracy belonging to optimal w':[test_acc_pot_opt,test_acc_pot_rec_opt ],
       'Test Accuracy Improvement':[test_acc_pot_improvement,test_acc_pot_rec_improvement],
       'Maximum test accuracy': [test_acc_pot_max,test_acc_pot_rec_max],
       'Number of potential points': [N_pot_max,N_pot_rec_max],}
  
  df = pd.DataFrame(data=d, index = index)
  return df



def highlight_confusion_matrix_difference(df):
  '''
  This function colors the confusion matrix.

  Input:
      df : difference of confusion matrix before and after retraining as dataframe
  Output:
         : same confusion matrix, but with colored elements. Positive diagonal elements are marked green, whereas negative elements are marked red.
           For off-diagonal elements the opposite is true.

  '''

  # Convert datafram to numpy:
  df_numpy = df.to_numpy()
  # Generate empty numpy array:
  a = np.full(df.shape, '', dtype='<U24')
  
  # Iterate over rows:
  for row in range(df.shape[0]):
    # Iterate over coloumns:
    for coloumn in range(df.shape[1]):
      # Extract element from numpy array: 
      diagonal_element = df_numpy[row,coloumn]
      # If diagonal element:
      if coloumn == row:
        if diagonal_element < 0:
          a[row,coloumn] = 'background-color: red'
        else:
          a[row,coloumn] = 'background-color: green'
      # If non diagonal element:
      else:
        if diagonal_element > 0:
          a[row,coloumn] = 'background-color: red'
        else:
          a[row,coloumn] = 'background-color: green'
          
  return pd.DataFrame(a, index=df.index, columns=df.columns)