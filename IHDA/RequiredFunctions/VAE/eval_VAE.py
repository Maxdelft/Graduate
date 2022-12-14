import torch
import numpy as np 
import matplotlib.pyplot as plt

def calc_risk_pc(model, x_sample, latent_dims,device):

  '''
    Function: This function is used to clalculate the "risk" whether a latent variable is facing posterior collapse. 

    Input:
       x_sample    := datasample used to map them to the latent space. Should be a torch.tensor with dtype = float32
       dir_model   := directory of the model
       latent_dims := latent dimension of model
    
    Output:
       
     '''

  # Sample data from dataset:
  x_sample = x_sample.to(device)

  # Calculate mean and variance: 
  mu, var    = calc_mu_var(device,model,x_sample)

  # Initialize lists to store statistics in: 
  post_collap_var,post_collap_risk           = [],[]
  non_post_collap_var, non_post_collap_risk  = [],[]

  for latent_var in range(latent_dims):
    post_collap, post_risk = check_post_collap(mu,var,latent_var)
    if post_collap == 1: 
      post_collap_var.append(latent_var)
      post_collap_risk.append(post_risk)
    elif post_collap == 0:
      non_post_collap_var.append(latent_var)
      non_post_collap_risk.append(post_risk)
  
  print('Potentially you are not facing posterior collapse for the following latent dimensions:')
  print(non_post_collap_var)
  print('The belonging risks for posterior collapse are: ')
  print(non_post_collap_risk)
  print('------------------------------------------------------------------------------------------------')
  print('Potentially you are facing posterior collapse for the following latent dimensions:')
  print(post_collap_var)
  print('The belonging risks for posterior collapse are: ')
  print(post_collap_risk)



def calc_mu_var(device,model,x_sample):
    '''
    Function: This function feeds x_sample through the decoder and outputs the resulting mean and variance representation.
    
    Input:
       model    := VAE model
       x_sample := features to map to latent space. Should be torch tensor with dtype = 32

    Output:   
       mu, var  := Mean and Variance representation of X
    '''
    z, mu, logvar  = model.encoder(x_sample.to(device))
    mu             = mu.detach().cpu().numpy()
    var            = torch.exp(logvar).detach().cpu().numpy()

    return mu, var


def check_post_collap(mu,var,latent_var):
  '''
  Function: This function calculates wether a certain function is facing posterior collapse or not. 
            This is done by checking how much of the data are mapped to variance of close to one.

  Hyperparameters:
    N_max     := upper bound of bin
    N_min     := lower bound of bin
    threshold := if threshold fraction of the data is mapped to variance between N_min and N_max, 
               it is assumed that the considered latent variable is facing posterior collapse
  
  Input: 
     mu:  mean of samples mapped to latent space
     var: variance of samples
     latent_var: considered latent variable
  
  Output:
     post_collap: if risk of post collapse greater than treshhold it will be set to one
     post_risk:   belonging risk of post collapse
  '''
  
  # Hyperparameter:
  N_max     = 1.1 # Max bin
  N_min     = 0.7 # Min bin
  threshold = 0.6 # Posterior collapse if threshold*100% of the total data falls into the bin.

  N_hist, hist = np.histogram(var[:,latent_var],bins = 20,range = (0,1.1))
  
  # Initialize post_collap and N_post:
  post_collap = 0
  N_post = 0 
  
  for i in range(len(N_hist)):
    if hist[i+1]>N_min and hist[i+1]<N_max:
      N_post = N_post + N_hist[i]

  post_risk = N_post/len(var)

  if post_risk > threshold:
    post_collap = 1

  return post_collap, post_risk 


def plot_hist(mu, var, lat_var):
  '''
     Function: This function plot two seperate histograms of mean and variance distribution.
     Input: 
           mu : mean
          var : variance 
          lat_var : considered latent variable
  '''
  # Set figure size:
  plt.rcParams["figure.figsize"] = (16,9)
  
  # Initialize figure:
  fig, axs = plt.subplots(1,2)

  # Plot figure
  fig.suptitle('Histogram belonging to latent variable:' +  str(lat_var))
  axs[0].hist(mu[:,lat_var],bins = 20)
  axs[0].title.set_text('Histogram of mu')
  axs[0].set_xlabel('mu')
  axs[0].set_ylabel('n')
  axs[1].hist(var[:,lat_var],bins = 20)
  axs[1].set_xlabel('variance')
  axs[1].set_ylabel('n')
  axs[1].title.set_text('Histogram of variance')



def plot_latent(x,y,model,lat_var_1,lat_var_2,n_plot):

  '''
  Function to plot latent space:

  Input:
     x,y         : features, labels
     model       : VAE
     lat_var_1/2 : latent variables that should used for plotting the latent space
     n_plot      : Number of samples considered for plotting the latent space
  '''

  device = 'cuda'
  classes   = ['Still', 'Walking', 'Run', 'Bike','Car','Bus', 'Train','Subway']
  cdict     = {0:'blue',1:'orange',2:'green',3:'red',4:'purple',5:'pink',6:'cyan',7:'gray'}
  
  # Set figure size:
  plt.rcParams["figure.figsize"] = (16,9)

  # Generate random index:
  idx = np.random.randint(0,len(x),n_plot)

  # Extract random index:
  x_latent = x[idx,:]
  y_latent =  y[idx]

  # Map to latent space:
  with torch.no_grad():
    Z , mu, logvar = model.encoder(x_latent.to(device))
  
  # Bring in useful form for plotting:
  Z      = Z.detach().cpu().numpy()
  Z1     = Z[:,lat_var_1].reshape(len(Z),1)
  Z2     = Z[:,lat_var_2].reshape(len(Z),1)
  Z      = np.concatenate((Z1,Z2),axis = 1)


  # Plot latent space:
  fig = plt.figure()
  ax = fig.add_subplot(111)
  for i in range(0,8,1):
    ix = np.where(y_latent == i)
    ax.scatter(Z[ix,0],Z[ix,1], c = cdict[i], label = classes[i], s = 30, alpha = 0.5)
    ax.legend(loc = 'best',markerscale = 5, fontsize = 25)
    ax.set_xlabel('Latent space variable $z$ ' + str(lat_var_1))
    ax.set_ylabel('Latent space variable $z$ '+ str(lat_var_2))
    #ax.set_xlim([-8,4])
    #ax.set_ylim([-6,6])
    plt.title('SHL dataset mapped to the latent space of a variational autoencoder')



def plot_x_rec_vs_x(x_rec,x_sample,y_sample):
  '''
  Function: Plots original versus reconstructed signal for VAE_1

  Input:
    x_rec    : reconstructed signal
    x_sample : original signal
    y_sample : sample 

  '''
  # Extract label
  classes    = ['Still', 'Walking', 'Run', 'Bike','Car','Bus', 'Train','Subway']
  label       = classes[int(y_sample)]
  
  # Calcuate frequency vector
  T = 1/100 # Sampling frequency
  N = 500   # Number of samples
  
  # Generate frequency vector.
  xf  = np.linspace(0.0, 1.0/(2.0*T), (N)//2+1)

  # Set figure size:
  plt.rcParams["figure.figsize"] = (16,9)

  # Plot reconstructe verus original sample
  fig, axs = plt.subplots(1,3)
  fig.suptitle('Signal versus its reconstruction for label: ' + label)

  axs[0].plot(xf, x_rec[:251],label = 'X_rec', color = 'orange',linewidth =4)
  axs[0].plot(xf, x_sample[:251],label = 'X', color = 'blue',linewidth =1.5)
  axs[0].legend()
  axs[0].title.set_text('Accelerometer')

  axs[1].plot(xf,  x_rec[251:502],label = 'X_rec', color = 'orange',linewidth =4)
  axs[1].plot(xf,  x_sample[251:502],label = 'X', color = 'blue',linewidth =1.5)
  axs[1].title.set_text('Gyro')
  axs[1].legend()

  axs[2].plot(xf, x_rec[502:753],label = 'X_rec', color = 'orange',linewidth =4)
  axs[2].plot(xf, x_sample[502:753],label = 'X', color = 'blue', linewidth  = 1.5 )
  axs[2].title.set_text('Magnetometer')
  axs[2].legend()