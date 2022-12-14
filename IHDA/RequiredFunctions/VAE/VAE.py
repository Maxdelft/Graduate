import torch.nn.functional as F
import torch
from torch import device, nn

class Encoder(nn.Module):
    def __init__(self,  latent_dims):
        super(Encoder, self).__init__()

        # Define convolutional layers:
        self.batch1  = nn.BatchNorm1d(500)
        self.batch2  = nn.BatchNorm1d(500)
        self.batch3  = nn.BatchNorm1d(500)

        # Define linear layers: 
        self.fc1     = nn.Linear(753, 500)
        self.fc2     = nn.Linear(500, 500)
        self.fc3     = nn.Linear(500, 500)
        self.fc4     = nn.Linear(500,latent_dims) 
        self.fc5     = nn.Linear(500,latent_dims) 

        # Define proportion or neurons to dropout:
        #self.dropout = nn.Dropout(dropout_ratio)

        # Requried variables for Variational Autoencoder 
        self.kl      = 0
        self.kl_mean = 0

    def forward(self, x):
        # Input layer
        x = x.view(-1,753) # dimension for 2d input [batch_size, features] / dimension for 3d input [batch_size,channels,number_features]

        # Hidden layers:
        x = F.leaky_relu(self.batch1(self.fc1(x)))
        x = F.leaky_relu(self.batch2(self.fc2(x)))
        x = F.leaky_relu(self.batch3(self.fc3(x)))
      

        mu     = self.fc4(x)
        logvar = self.fc5(x)

        # Calculate latent variable z:
        z            = torch.randn_like(mu)*torch.exp(0.5*logvar) + mu

        # Calculate kl_loss:
        self.kl      = 0.5*torch.mean((torch.sum(torch.exp(logvar) + mu**2-logvar-1,dim = 1)))
  
        return z, mu, logvar


###################################### Decoder ############################################

class Decoder(nn.Module):
    def __init__(self, latent_dims,dropout_ratio):
        super(Decoder, self).__init__()
       
        self.latent_dims = latent_dims

        # Linear layers:
        self.fc1 = nn.Linear(latent_dims,500)
        self.fc2 = nn.Linear(500,500)
        self.fc3 = nn.Linear(500,500)
        self.fc4 = nn.Linear(500,753)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, z):
        
        z = z.view(-1,self.latent_dims) # [batch_size, features]
         
        # Linear layers 
        x = F.leaky_relu(self.fc1(z)) 
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc4(x))

        # Bring x into the same shape as the output:
        x = x.view(-1,753)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, dropout_ratio):
        super(VariationalAutoencoder,self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims, dropout_ratio)
    
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x = self.decoder(z)
        return x



def initialize_VAE_1(VAE_1, dir_fc_base):
    ''' 
    Function: Initializes weights of VAE_1 with the weights of baseline classifier.
    Input:  
         VAE_1       : VAE_1 with random weights.
         dir_fc_base : direcotry of baseline classifier
    Output: 
         VAE_1 : VAE_1 with decoder weights equal to the weights of the baseline classifier. 
                 Furthermore those weights are fixed.
    '''
    
    checkpoint_fc = torch.load(dir_fc_base)

    # Initialize VAE weights:
    VAE_1.encoder.fc1.bias.data   = checkpoint_fc['model_state_dict']['fc1.bias'].data
    VAE_1.encoder.fc1.weight.data = checkpoint_fc['model_state_dict']['fc1.weight'].data
    VAE_1.encoder.fc1.requires_grad_(False)
    
    VAE_1.encoder.fc2.bias.data   = checkpoint_fc['model_state_dict']['fc2.bias'].data
    VAE_1.encoder.fc2.weight.data = checkpoint_fc['model_state_dict']['fc2.weight'].data
    VAE_1.encoder.fc2.requires_grad_(False)
    
    VAE_1.encoder.fc3.bias.data   = checkpoint_fc['model_state_dict']['fc3.bias'].data
    VAE_1.encoder.fc3.weight.data = checkpoint_fc['model_state_dict']['fc3.weight'].data
    VAE_1.encoder.fc3.requires_grad_(False)
    
    VAE_1.encoder.batch1.bias.data                = checkpoint_fc['model_state_dict']['batch1.bias'].data
    VAE_1.encoder.batch1.num_batches_tracked.data = checkpoint_fc['model_state_dict']['batch1.num_batches_tracked'].data
    VAE_1.encoder.batch1.running_var.data         = checkpoint_fc['model_state_dict']['batch1.running_var'].data
    VAE_1.encoder.batch1.running_mean.data        = checkpoint_fc['model_state_dict']['batch1.running_mean'].data
    VAE_1.encoder.batch1.requires_grad_(False)
    
    VAE_1.encoder.batch2.bias.data                = checkpoint_fc['model_state_dict']['batch2.bias'].data
    VAE_1.encoder.batch2.num_batches_tracked.data = checkpoint_fc['model_state_dict']['batch2.num_batches_tracked'].data
    VAE_1.encoder.batch2.running_var.data         = checkpoint_fc['model_state_dict']['batch2.running_var'].data
    VAE_1.encoder.batch2.running_mean.data        = checkpoint_fc['model_state_dict']['batch2.running_mean'].data
    VAE_1.encoder.batch2.requires_grad_(False)
    
    VAE_1.encoder.batch3.bias.data                = checkpoint_fc['model_state_dict']['batch3.bias'].data
    VAE_1.encoder.batch3.num_batches_tracked.data = checkpoint_fc['model_state_dict']['batch3.num_batches_tracked'].data
    VAE_1.encoder.batch3.running_var.data         = checkpoint_fc['model_state_dict']['batch3.running_var'].data
    VAE_1.encoder.batch3.running_mean.data        = checkpoint_fc['model_state_dict']['batch3.running_mean'].data
    VAE_1.encoder.batch3.requires_grad_(False)

    return VAE_1