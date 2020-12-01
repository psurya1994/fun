## Analysis of how representations evolve during learning


### Summary

The goal of this project is to visualize how the representations evolve over time (visualized using PCA(n_comp=2) of the training datapoints) when we train a toy dataset with supervised learning, SimCLR, BYOL, etc.,. 

Comparing the evolution of representations might give us some insights into why these algorithms work so well.

### Results

When I train the SimCLR loss, I don't actually see representations become linearly separable in my toy experiments. Which is surprising. Even after spending several hours, I wasn't able to figure out why.

Methodology: I use the dataset shown in the image. All red points considered as data augmentation of the sample 1 and all blue points as data augmentation of sample 2. 

