import numpy as np

class UsefulFunctions:

  def normalize(self, X):
    """
    Implements z-score normalization for a matrix

    Arguments:
    X (matrix) : matrix to be z-score normalized by feature
    
    Returns: 
    X_norm (matrix) : normalized matrix by feature
    mean (vector) : mean along every feature
    std (vector) : standard deviation along every feature 
    """
    mean = np.mean(X, axis=0)
    std = np.maximum(np.std(X, axis=0), 1e-15)
    X_norm = (X - mean) / std
    return X_norm, mean, std

  def train_test_split(self, X, ratio):
    """
    Performs a test-train split for the given ratio

    Arguments:
    X (matrix) : Dataset to be splitted
    ratio (scalar) : ratio to be splitted

    Returns:
    X_train (matrix) : train data
    X_test (matrix) : test data
    """
    size = int(X.shape[0]*ratio)
    X_train = X[0:size]
    X_test = X[size:]
    return X_train, X_test

  def one_hot(self, y, classes):
    # Performs one hot encoding for given y and number of classes
    return np.eye(classes)[y]

  def reverse_one_hot(self, y):
    # Reverses the one hot encoding performed
    return np.argmax(y, axis=1)

  def add_bias(self, X):
    # Adds a feature of 1s
    return np.c_[np.ones(X.shape[0]), X]

  def r2_score(self, true, pred):
    """
    Calculates r2_score for given true and predicted values

    Arguments:
    true (vector) : true values
    pred (vector) : predicted values

    Returns: 
    score (scalar) : R2 score of the given data
    """
    res = true - pred
    square = true - np.mean(true)
    score = 1 - ((np.sum(np.square(res)))/(np.sum(np.square(square))))
    return score

  def mean_squared_error(self, true, pred):
    # Calculates mean squared error of model
    return np.mean((true - pred)**2)

  def mean_absolute_error(self, true, pred):
    # Calculates mean absoulte error of model
    return np.mean(np.abs(true - pred))

  def classification_accuracy(self, true, pred):
    """
    Displays the accuracy of a classification model

    Arguments"
    true (vector) : true values
    pred (vector) : predicted values
    """
    accurate_predictions = np.sum(true == pred)
    accuracy = np.mean(true == pred)
    print(f"Successfully predicted {accurate_predictions} out of {int(len(true))}")
    print(f"Accuracy: {accuracy}")