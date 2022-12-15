1. The script_baseline_classifier_fc.ipynb is used to train the fully connected baseline classifier according to: https://www.researchgate.net/publication/328679101_Benchmarking_the_SHL_Recognition_Challenge_with_Classical_and_Deep-Learning_Pipelines
2. In this file the variable *dir_IHDA* needs to be specified. It represents the directory of the folder *IHDA*.
4. If training the classifier should be performed, the dataset stored in folder */IHDA/SHLData/2018/Features/* needs to be splitted into training and validation dataset. This will be done if the variable *new_dataset_state* is set to one. It needs to be done at least once!
5. The data used to train the baseline classifier will be stored in the file /IHDA/BaselineClassifier/Classifier/Data
6. If the classifier should be retrained on upsampled dataset in which all classes are represented by an equal amount of samples set the variable *upsample_state* equal to one.
7. The final model and the one that achieves the lowest training validation accuracy will be saved in: /IHDA/BaselineClassifier/Classifier as final_model.pth and best_model.pth
