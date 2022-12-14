import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import scipy
import matplotlib.pyplot as plt




def map_dataloader_to_latent(dataloader,model,device):
  ''' 
  Function: This function maps input to latent space

  Input:
    dataloader: dataloader !!!!!ATTENTION!!!!! DATALOADER SHOULD NOT BE SHUFFLED
    model: vae model 
    device: device to perform calculations on

  Output:
     Z = latent space representation of x stored in numpy array.
     Y = labels
  ''' 


  # List to store latent space vector Z:
  Z = [] 

  # Feed dataset through the encoder. 
  # Needs to be done by makeing use of dataloader, since otherwise the system crashes
  for i, data in enumerate(dataloader,0):
    # Load data
    X_batch , Y_batch = data                         
    X_batch = X_batch.to(device)
    
    with torch.no_grad():
      z,mu,logvar = model.encoder(X_batch)

    z = z.detach().cpu()
    Z.append(z)

  Z = torch.vstack(Z)
  
  return Z

def map_dataloader_to_rec(dataloader,model,device):
  ''' 
  Function: This function maps input to its reconstruction

  Input:
    dataloader: dataloader !!!!!ATTENTION!!!!! DATALOADER SHOULD NOT BE SHUFFLED
    model: vae model 
    device: device to perform calculations on

  Output:
     X_rec = reconstructed signal
     Y     = labels
  ''' 


  # List to store latent space vector Z:
  X_rec = [] 

  # Feed dataset through the encoder. 
  # Needs to be done by makeing use of dataloader, since otherwise the system crashes
  for i, data in enumerate(dataloader,0):
    # Load data
    X_batch , Y_batch = data                         
    X_batch = X_batch.to(device)
    
    with torch.no_grad():
      x_rec = model(X_batch)

    x_rec = x_rec.detach().cpu()
    X_rec.append(x_rec)

  X_rec = torch.vstack(X_rec)
  
  return X_rec


def split_potential(X,Y,Z,potential,device):
    '''
    Function: This function splits X,Y,Z in two arrays, where one contains all points with postivie potential 
    and the other all points with negatvie potential.

    Input:
      X, Y, Z    : features, labels, latent space representation
      potential_ : potential of each point

    Output:
      X_pot, Y_pot, Z_pot: features, labels, and latent space representation of points with positive potential
      X_not_pot, Y_not_pot, Z_not_pot features, labels, and latent space representation of points with negative potential
    '''
    # Extract index of points with positive and negatvie potential:
    idx_pot     = torch.where(potential > 0)[0]
    idx_not_pot = torch.where(potential <=0)[0]
    
    # Map to device:
    idx_pot, idx_not_pot = idx_pot.to(device), idx_not_pot.to(device)
    Z, Y, X = Z.to(device), Y.to(device), X.to(device)

    # Extract points with positiv potential
    Z_pot       = Z[idx_pot,:]
    Y_pot       = Y[idx_pot]
    X_pot       = X[idx_pot,:] 
    
    # Extract point with negative potential:
    Z_no_pot   = Z[idx_not_pot,:]
    Y_no_pot   = Y[idx_not_pot]
    X_no_pot   = X[idx_not_pot,:]
    
    return X_pot, Y_pot, Z_pot, X_no_pot, Y_no_pot, Z_no_pot


def plot_latent_space_with_pot(Y_pot, Z_pot,Y_no_pot, Z_no_pot,N_pot, N_no_pot,lat_var_1,lat_var_2):
    '''
    Function: Plot latent space representation of including points with positive potential
    Input:
       Y_pot, Z_pot         : Labels and latent space representation of points with positive potential
       Y_no_pot, Z_no_pot   : Labels and latent space reperesentation of points with negative potential
       N_pot                : Number of points to plot with potential
       N_no_pot             : Number of points to plot with no potential
       lat_var_1, lat_var_2 : latent variable 1 and 2 to plot
    ''' 
    
    # Extract points with no potential
    idx_no_pot_plot = np.random.randint(0,len(Z_no_pot),N_no_pot)
    Z_no_pot_plot   = Z_no_pot[idx_no_pot_plot,:].numpy()
    Y_no_pot_plot   = Y_no_pot[idx_no_pot_plot].numpy()
    
    # Extract points with potential:
    idx_pot_plot   = np.random.randint(0,len(Z_pot),N_pot)
    Z_pot_plot     = Z_pot[idx_pot_plot].numpy()

    classes   = ['Still', 'Walking', 'Run', 'Bike','Car','Bus', 'Train','Subway']
    cdict     = {0:'blue',1:'orange',2:'green',3:'red',4:'purple',5:'pink',6:'cyan',7:'gray'}
    
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    for i in range(0,8,1):
        ix = np.where(Y_no_pot_plot == i)
        ax.scatter(Z_no_pot_plot[ix,lat_var_1],Z_no_pot_plot[ix,lat_var_2], c = cdict[i], label = classes[i], s = 25, alpha = 0.5)
    ax.scatter(Z_pot_plot[:,lat_var_1],Z_pot_plot[:,lat_var_2],label = 'Potential',marker ='o', s= 30, alpha = 0.7, color = 'lime' )
    ax.legend(loc = 'best',markerscale = 3)
    #ax.set_xlim([-6,6])
    # #ax.set_ylim([-6,6])
    ax.legend(loc = 'best',markerscale = 4, fontsize = 25)
    plt.title('SHL dataset mapped to the latent space of a variational autoencoder')
    plt.show()


def main_potential(Z,Y,w,gamma,dist_metric,device):
    '''
    Function: Main function to calculate potential

    Input: 
         Z           : latent space representation
         Y           : label
         w           : neighbourhood hyperparemter
         gamma       : hyperparameter potential
         dist_metric : dist_metric ### ATTENTION IF COS, THERE ARE TWO THINGS THAT NEED TO BE CONSIDERED
         device      : device on which calculations are performed 
    
    Output:
        potential: potential of each point

    '''
    Z = Z.to(device)
    Y = Y.to(device)
  
    # Initialize list to store potential values:
    potential = []

    # Iterate through Z
    for idx in range(0,len(Z)):
      
      # Print progress information
      if idx%10000 == 0:
        print('Progress: ' + str(idx) + '/' + str(len(Z)))
      
      # Extract Z_i for which the potential is calculated
      Z_i = Z[idx,:]
      Y_i = Y[idx]

      # Calculate distance:
      dist = dist_metric(Z_i,Z)
      
      # Find index of neighbouring samples
      '''ATTENTION: ONLY FOR COSINE dist > w, otherwise dist < w'''
      idx_neigh = torch.where(dist > w)[0] # dist > w is used in combination with cosine similiarty function. Using cosine similiarty two signals are most similiar if it reaches its maximimum values 1

      # Extract neighbouring labels:
      Z_neigh = Z[idx_neigh,:]
      Y_neigh = Y[idx_neigh]
      if Y_i not in Y_neigh:
        potential_i = 0
      else:
        idx_different_label = torch.where(Y_i != Y_neigh)[0]
        idx_same_label      = torch.where(Y_i == Y_neigh)[0]
        
        # Spllit latent space representations
        Z_neigh_different = Z_neigh[idx_different_label,:]
        Z_neigh_same      = Z_neigh[idx_same_label,:]
        
        p_different = torch.sum(torch.exp(torch.sum(-torch.sub(Z_neigh_different, Z_i)**2,dim = 1)/gamma))
        p_same      = torch.sum(-torch.exp(torch.sum(-torch.sub(Z_neigh_same, Z_i)**2,dim = 1)/gamma)) 

        # Only minus one by using cosine similairty matrix. Compensation for calculating the distance too itself. 
        ''''ATTENTION: only for cosine, otherwise p_same = p_same'''
        p_same = p_same + 1

        potential_i = p_same + p_different
      potential.append(potential_i)
    
    potential = torch.vstack(potential)
    return potential