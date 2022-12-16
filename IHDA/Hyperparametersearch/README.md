In this folder all jupyter notebook scripts and corresponding trained models are stored that are required to perform the hyperparameter search. 
1. The first hyperparametersearch performed is the one to select a value for beta, which represents the penalization of the Kullback-Leibler divergence in the loss function of the VAE.
All coresponding files are found in the folder *Graduate/IHDA/Hyperparametersearch/beta_VAE*. Therefore, to perform the hyperparameter search for beta_VAE open one of the files 'hyperparametersearch_beta_VAE_x.ipynb' in the folder
*Graduate/IHDA/Hyperparametersearch/beta_VAE*. The *x* represents one of the layers [0,1,2] that needs be optimized.
