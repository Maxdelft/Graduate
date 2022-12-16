import matplotlib.pyplot as plt


def plot_losses_seperatly(history):
  '''
  Function: Plots train and validation reconstruction loss and kullback leibler divergence seperatly

  Input:
    history: dictonary that should contain training results specified as train_rec_loss, valid_rec_loss, train_kl_loss and valid_kl_loss
  '''

  # Initialize figure:
  fig, axs  = plt.subplots(1,2)
  
  # Extract train and valid reconstruction losses:
  train_rec_loss = [x['train_rec_loss'] for x in history]
  valid_rec_loss  = [x['valid_rec_loss'] for x in history]
  
  # Plot reconstruction losses:
  axs[0].plot(train_rec_loss,label = 'train')
  axs[0].plot(valid_rec_loss,label = 'valid')
  axs[0].legend()
  axs[0].set_xlabel('Epochs')
  axs[0].set_ylabel('Reconstruction loss')

  # Extract train and valid kl loss:
  train_kl_loss = [x['train_kl_loss'] for x in history]
  valid_kl_loss  = [x['valid_kl_loss'] for x in history]
  
  # Plot train and valid kl loss:
  axs[1].plot(train_kl_loss,label = 'train')
  axs[1].plot(valid_kl_loss,label = 'valid')
  axs[1].legend()
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel('KL Divergence')
  plt.show()
  plt.close()


def plot_classifier_losses_seperatly(history):
  '''
  Function: This function plots train, test and evaluation loss of classifier aswell as evolution of accurcy 
  Input: 
      history: dictonary that should containing training results. The following values should be stored: train_loss, valid_loss, train_acc and valid_acc.
  '''

  # Initialize plot:
  fig, axs  = plt.subplots(1,2)
  
  # Extract losses:
  train_loss = [x['train_loss'] for x in history]
  valid_loss  = [x['valid_loss'] for x in history]
    
  # Plot losses:
  axs[0].plot(train_loss,label = 'train')
  axs[0].plot(valid_loss,label  = 'valid')
  axs[0].legend()
  axs[0].set_xlabel('Epochs')
  axs[0].set_ylabel('Loss')

  # Extract accuricies:
  valid_acc = [x['valid_acc'] for x in history]
  train_acc = [x['train_acc'] for x in history]
  
  # Plot accuracies:
  axs[1].plot(train_acc,label = 'train')
  axs[1].plot(valid_acc,label = 'valid')
  axs[1].legend()
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel('Accuracy')
  plt.show()
  plt.close()

