For every layer of the baseline classifier, a Variational Autoencode(VAE) is according to the IHDA method. The scripts for training the VAE can be 
found in folder 'VAE_$layer' with layer equal to 1,2 or 3. In those script the directory of the file IHDA needs to be specified. The trained models will be
or are stored in the folder 'VAE_$layer/Models/VAE_beta_$beta_latent_dims_$latent_dims_epochs_$epochs', where $beta represents beta value of the VAE,
$latent_dims the latent dimensions and $epochs the number of training epochs. The final model and the model with the lowest validation loss are stored in those folders
as final_model.pth and best_model.pth.
