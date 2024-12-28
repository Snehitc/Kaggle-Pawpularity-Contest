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

# Ensemble Set
A list of models used in each set is shown below.
![Ensemble_Set](Images/Ensemble_set.jpg)


# Key Packages
>## SVR: ```cuML``` vs ```sklearn```
>[![cuML](https://img.shields.io/badge/cuML-purple?logo=Rapids
)](https://docs.rapids.ai/api/cuml/stable/api/) does the SVR calculation faster (using GPU accelerators). Note that while experimenting, it only worked with the ```Tesla T4``` GPU.
>If ```cuML``` doesn't work, then use [![sklearn](https://img.shields.io/badge/sklearn-white?logo=scikit%20learn
)](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVR.html) to compute SVR.

>## scipy.optimize
>To find the optimized combination of the ensemble set, a ```minimize``` package from the ```optimize``` library of ```scipy``` is used.\
>Link: [![minimize](https://img.shields.io/badge/scipy-optimize-yellow?logo=scipy
)](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)


# Note
>## Execution
>Use the kaggle notebook for execution: [![notebook](https://img.shields.io/badge/-Notebook-lightblue?logo=kaggle
)](https://www.kaggle.com/code/snehitchunarkar/ensembleset-svr-for-pawpularityscore) \
>Select the ```Copy & Edit``` option; it'll create a copy of the code with the necessary input files (which are not added to GitHub).

>## Saved embeddings
>To save time in training, the output from the ensemble set's models is saved in  ```.npz``` files. If required to use the models for training data, make the necessary changes in the ```Specify Inputs (all models)``` section.

>## Best Score
>![Score](Images/Pawpularity5_V180.PNG)

# Reference
[![Refer](https://img.shields.io/badge/Reference-Notebook-white?logo=kaggle
)](https://www.kaggle.com/code/titericz/imagenet-embeddings-rapids-svr-finetuned-models) \
The above notebook by ```Giba``` on Kaggle is helpful for this work.
