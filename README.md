# deep-learning-challenge

# Overview
The objective of this project is to create a binary classifier that predicts the probability of applicants achieving success after receiving funding from Alphabet Soup. To accomplish this, we will utilize the features provided in the dataset and employ various machine learning techniques to train and evaluate the performance of the model. Our goal is to optimize the model and achieve an accuracy score of over 75%.

# Results
## Data Preprocessing
The primary goal of the model is to predict the likelihood of success for applicants who receive funding. This is determined by the "IS_SUCCESSFUL" column in the dataset, which serves as the target variable. The feature variables encompass all columns except for the target variable and non-relevant variables like EIN and names. These features capture essential information about the data and play a crucial role in predicting the target variable. Non-relevant variables, which neither serve as targets nor features, are removed from the dataset to prevent potential noise that could confuse the model.

During the preprocessing stage, I employed binning or bucketing techniques to handle rare occurrences in the "APPLICATION_TYPE" and "CLASSIFICATION" columns. Subsequently, I utilized one-hot encoding to transform categorical data into numerical data. The dataset was then divided into separate sets for features and targets, as well as for training and testing purposes. Finally, I applied data scaling to ensure uniformity in the distribution of the data.

## Compiling, Training, and Evaluating the Model
For my initial model, I opted for a three-layer architecture: an input layer with 80 neurons, a second layer with 30 neurons, and an output layer with 1 neuron. I chose this configuration to ensure that the total number of neurons in the model was around 2-3 times the number of input features. In this specific case, there were 43 input features left after eliminating 2 irrelevant ones.

To facilitate binary classification, I employed the relu activation function for the first and second layers, while the sigmoid activation function was chosen for the output layer.

During the training phase, I trained the model for 100 epochs and obtained an accuracy score of approximately 74% for the training data and 72.9% for the testing data. There were no discernible signs of overfitting or underfitting observed.

## Optimization attempts
In my final optimization attempt, I made additional adjustments to improve the model's performance. Despite these efforts, the final optimization attempt resulted in an accuracy score of approximately 72.8%. Unfortunately, I was not able to achieve the desired goal of a 75% accuracy score even after several attempts to optimize the model.

# Summary
Since the target accuracy of 75% was not achieved with the models mentioned above, I would not recommend any of those models. However, if given additional time, I would explore alternative approaches to improve the performance. One possibility would be to incorporate the Random Forest Classifier and experiment with different preprocessing modifications.

To further optimize the model, I would consider making changes to the dropout layers by adjusting their rates or removing them altogether. Additionally, I would explore different activation functions and experiment with varying the number of layers and neurons in the model architecture.

By exploring these alternative approaches and fine-tuning the model, I believe there is a potential to achieve the desired goal of 75% accuracy.
