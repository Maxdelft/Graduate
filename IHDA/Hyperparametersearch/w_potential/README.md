# Results of hyperpametersearch for distance parameter w:

In the following section, the main results of the hyperparameter search of the distance parameter *w* will be highlighted. *w* represents the distance between 
two samples. If the coresponding distance is smaller than *w* the sample will be considered for calculating the potential of a certain sample.

### Neighbourhood:
First, the potential is a property of every sample in the training dataset. The goal of the potential is to identify samples that are surrounded in their neighbourhood by more samples belonging to a different class than of its own. Therefore, first a definition of the neighbourhood of a sample is requried.

### Neighbourhood

The authors of the [IHDA](https://proceedings.neurips.cc/paper/2020/file/074177d3eb6371e32c16c55a3b8f706b-Paper.pdf) algorithm define the neighbourhood of a sample as follows:
![performance_stats_BC](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/neighbourhood.png)

![performance_stats_BC](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/Potential.png)

![performance_stats_BC](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/neighbourhood.png)
### Performance of baseline classifer: 
Let's first have a look at the performance of the baseline classifier, which are summarized below:
![performance_stats_BC](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/BaselineClassifier/performance_stats_BC.png)

Furthermore, lets take a look at the confusion matrix shown in the following image:
![confusion_matrix_BC](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/BaselineClassifier/confusion_matrix_BC.png)

Following observations should be made:
  * More than 85% of the data belonging to the transportation modes *Still, Walk, Bike* and *Run* are correctly classified.
  * 30% of the data belonging to the transportation mode *Car* is missclassified as *Bus*
  * 33% of the data belonging to the transportation mode *Train* is missclassified as *Subway*

### Hyperparametersearch w:

For different values of *w* the potential set is calculated. Lets analyze the different potential sets first. In the following image some main results are listed:

![potential_stats](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/potential_stats.png)

The follwoing observations should be made:
* with increasing value for *w* the number of samples with postivie potential decreases
* up to a value of *w* equal to 0.95 the size the number of points with positive potential almost does not decrease
* up to a value of *w* equal to 0.95 the distribution of the classes in dataset of points with positive potential is approximatley as follows: *Still ~ 22%, Walking ~2.5%, Run ~0.2%, Bike ~4%, Car ~ 20%, Bus ~ 19%, Train ~ 18%* and *Subway ~ 12%*
* up to a value of *w* equal to 0.990 and 0.9991 the following changes of the distribution should be highlighted: *Still ~10% (-12%), Bus ~ 22% (+3%), Train ~ 24% (+6%)* and *Subway ~ 16 (+4%)*
* and lastly, for values of *w* between *0.9994* and *0.9998* the distribution of the different classes varies


For visualization the 2d latent space representation of the VAE is plotted. The points with positive potential are marked lime green. The following potential points belong to *W=0.1* :
![ls_w_0.1](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/ls_w_0.1.png)

whereas the following potential points belong to *w=0.9998*: 
![ls_w_0.9998](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/ls_w_0.9998.png)
