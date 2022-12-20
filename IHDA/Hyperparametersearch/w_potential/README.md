# Results of hyperpametersearch for distance parameter w:

In the following section, the main results of the hyperparameter search of the distance parameter *w* will be highlighted. *w* represents the distance between 
two samples. If the coresponding distance is smaller than *w* the sample will be considered for calculating the potential of a certain sample.

### Neighbourhood:
First, the potential is a property of every sample in the training dataset. The goal of the potential is to identify samples that are surrounded in their neighbourhood by more samples belonging to a different class than of its own. Therefore, first a definition of the neighbourhood of a sample is requried.

### Neighbourhood

The authors of the [IHDA](https://proceedings.neurips.cc/paper/2020/file/074177d3eb6371e32c16c55a3b8f706b-Paper.pdf) algorithm define the neighbourhood *N_p* of a sample *p* as follows:
![performance_stats_BC](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/neighbourhood.png)

Speaking, all samples *q* for which the distance between them and the considered sample *p* is smaller than *w* are neighbouring samples of the sample *p* and therefore belong to the neighbourhood *N_p*. It should be observed that the definition of the neighbourhood *N_p* requries a definition of the of the distance between two samples. It is reported in the paper [IHDA](https://proceedings.neurips.cc/paper/2020/file/074177d3eb6371e32c16c55a3b8f706b-Paper.pdf) that the cosine similarity delivers the best results.

#### Cosine Similarity

The cosine similarity is described mathematically as the division between the dot product of vectors and the product of the euclidean norms or magnitude of each vector. The similarity measurement is bounded between -1 and 1 and represents a measure of the cosine of the angle between the two non-zero vectors *A* and *B*. If the angle between two vectors is 90 degrees, the cosine similairty will have a value of 0, which means that two vectors are orthogonal or perpendiciular to each other. As the cosine similarity measurement gets closer to 1, then the angle between the two vectors A and B becomes smaller.

Accordingly, the more similar two samples are, the smaller the angle between those two samples and accordintly the higher the value for the cosine similarity. It should be observed that in opposite to most other similarity metrices, for this metrics it holds that the more similar two samples are the higher the value of the similairty metrics. Accordingly, using the cosine similairty the sample *q* belongs to the neighbourhood of *p* if  *dist(p,q) >= w*.
![Cosine Similairty](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/cosine_similairty.png)

#### Cosine Similarity Applied to IHDA
In the following the 2d latent space representation of a subset of the training data is plotted. Furthermore one sample is extacted and the Cosine Similarity to the plotted subset is calculated. The 400 samples with the highest (black) and lowest (olivegreen) similarity metric are marked in the laten space representation. Visually it can be observed that samples with high cosine similarity are close the considered sample, whereas samples with a low cosine similarity are further away.

![Cosine Similairty](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/Z_i_cs.png)

#### 


### Potential

After having defined which samples *q* belong to the neighbourhood of the sample *p*, calculating the potential of sample *p* can be introduced. The potential of sample *p* is calculated by iterating through all samples contained in the the defined neighbourhood.

For every sample *q* in the neighbourhood of the sample *p* the following calculation is made:
![Potential](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/e_potential.png)

The higher the difference between the samples *p* and *q*, the higher the value gets.Furthermore, gamma represents a hyperparameter that can be used to influence the slope of the function. The authors of the [IHDA](https://proceedings.neurips.cc/paper/2020/file/074177d3eb6371e32c16c55a3b8f706b-Paper.pdf) algorithm have fixed gamma to a value of gamma = 0.05. 

The potential of sample *p* is calculated by iterating through the neighbourhood of the considered sample. For every neighbouring sample *q* the 'e to the power' value is calculated. It is is added to the potential value of *p* if the samples *p* and *q* are of different class or substracted if they are of the same class. Finally, the potential of sample *p* represents a measure that reflects by how many samples of different class/ same class the considered sample is surrounded. The [IHDA](https://proceedings.neurips.cc/paper/2020/file/074177d3eb6371e32c16c55a3b8f706b-Paper.pdf) algorithm assumes that if a sample *p* has positive potential, it is surrounded by more samples of different class than of same class and therefore retraining the classifier on this sample should be performed.

The algorithm to calculate a samples potential can be described as follows:
![Potential](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/Potential.png)

#### Thoughts on calculating the potential

In the following I will briefly elobrate on some thoughts regarding the potential step. 
Intuitvley I would say that, if two samples are of different class, but they are close the value added to the potential should be high. In very close neighbours of different class are penalized more than samples of different class that are furhter away. The opposite should hold for substracting a value from the potential, if they are of the same class.

Now lets take a close look of the 'dynamics' of:
![Potential](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/e_potential.png)

If two samples are 'far' away from each other this value becomes large. Accordingly it behaves opposite to what I would intutivley do.
Therefore would it not make sense to use a penalization that behaves similar to: e^(1/(||p-q||))?

To clarify the critism on the current way of calculating the potential two situations are sketched for which the current algorithm would calulate a positive potential (case 1) and a negative potential (case 2).
![Potential](https://github.com/Maxdelft/Graduate/blob/main/IHDA/Images/Results/w_potential/critisim_on_potential.png)

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
