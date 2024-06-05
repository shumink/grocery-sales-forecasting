# Demand Forcast system

This repo implements a simple, batch-based demand forcast system based on historical sales record. This repo is structured as:
* `EDA.ipynb` contains the exploratory data analysis report.
* `train.ipynb` contains code that 
    1. fetches the training data and the feature,
    1. trains a light gbm model,
    1. deploy the model.
* `predict.ipynb` contains code that uses the deployed model to make a prediction.
* `features.py` contains feature services that can join features onto the training set.
* `dataset/` input data to the ML system, it should contain the training data and other feature tables that can join onto the training data in `.csv.7z` formats.
* `model/` contains different versions of the output model artifact.
