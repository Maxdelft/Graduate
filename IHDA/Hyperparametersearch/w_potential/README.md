# Results of hyperpametersearch for distance parameter w:

In the following section, the main results of the hyperparameter search of the distance parameter *w* will be highlighted. *w* represents the distance between 
two samples. If the coresponding distance is smaller than *w* the sample will be considered for calculating the potential of a certain sample.

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
