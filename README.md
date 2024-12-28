[![Contest](https://img.shields.io/badge/contest-Pawpularity-brightgreen?logo=kaggle
)](https://www.kaggle.com/competitions/petfinder-pawpularity-score)
[![notebook](https://img.shields.io/badge/-Notebook-lightblue?logo=kaggle
)](https://www.kaggle.com/code/snehitchunarkar/ensembleset-svr-for-pawpularityscore)

# Task Description
In the competition, we are given images (of cats and dogs) and a file containing respective metadata. Using this, we need to predict the Pawpularity score for the picture of a cat or dog. A brief idea is shown in the figure below. \
![Task](Images/Task_description.jpg)

# Architecture
The overall architecture for the task contains multiple ensemble sets, each with multiple models. Where all the models are frozen and only used to extract the image embedding from their output layers. The SVR is trainable, and the minimised function tunes the final output for optimized results. \
![Architecture](Images/architecture.jpg)

## SVR: ```cuML``` vs ```sklearn```
```cuML``` does the SVR calculation faster (using GPU accelerators). Note that while experimenting, it only worked with the ```Tesla T4``` GPU.
If ```cuML``` doesn't work, then use ```sklearn``` to compute SVR.

## scipy.optimize
[![minimize](https://img.shields.io/badge/scipy-optimize-yellow?logo=scipy
)](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)
To find the optimized combination of the ensemble set, a ```minimize``` package from the ```optimize``` library of ```scipy``` is used.


# Ensemble Set
A list of models used in each set is shown below.
![Ensemble_Set](Images/Ensemble_set.jpg)
