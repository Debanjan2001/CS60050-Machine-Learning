# CS60050 - MACHINE LEARNING ASSIGNMENT - 2: NAIVE BAYES CLASSIFIER

- This file contains the instructions to setup this project on your system.
- It also contains the brief overview about the additional files generated when you execute the code. 
- Please go through it once to get familiarised with the meaning of those additionally generated files.

## Authors:
1. Aaditya Agrawal (19CS10003)
2. Debanjan Saha (19CS30014)

### Project Setup for Linux OS
- Install jupyter notebook from https://jupyter.org/install.

- Install Anaconda from https://docs.anaconda.com/anaconda/install/index.html. (All the required libraries come pre-installed with conda)

### Instructions To Run the Code
- Run analysis_and_featureMatrix.ipynb to generate the feature matrix.

- Run training_and_testing.ipynb to run the classifier and get the test results.

### Testing takes approximately 40 seconds for a 70-30 split of train.csv

## Directory Structure:

#### ./dataset

- train.csv : Dataset for training and testing our classifier

#### ./

- Assignment - 2 Report.pdf : Report for our assignment.
- analysis_and_featureMatrix.ipynb : Analysis of the dataset and generates the feature Matrix.
- training_and_testing.ipynb : Implementation of Naive Bayes Classifier and also generates the statistics asked in question.

## Files generated after running the code:

- word_index.json : A json storing mapping of words to their corresponding column index in feature matrix.
- feature_matrix.npz : Feature Matrix