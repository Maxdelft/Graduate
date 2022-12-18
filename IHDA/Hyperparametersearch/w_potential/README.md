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
