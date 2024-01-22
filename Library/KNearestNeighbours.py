import UsefulFunctions as uf
import numpy as np

class KNearestNeighbours:
  def __init__(self):
    self.mean = 0
    self.std = 1

  def distances(self, x1, X2):
    """
    Returns the distances of one point (x1[vector]) wrt multiple points (X2[matrix])
    """
    return np.sum(np.abs(X2 - x1), axis=1)

  def compute(self, X_train, y_train, X_test, k=5, normalize=True):
    """
    Implements the KNN algorithm for the given data

    Arguments:
    X_train (matrix) : train data
    y_train (vector) : train data
    X_test (matrix) : test data
    k (scalar) : number of neighbours to be taken into consideration
    normalize (boolean) : indicates if data is needed to be normalized

    Returns: 
    predictions (vector) : predictions made using KNN algorithm
    """
    if(normalize):
      X_train, self.mean, self.std = uf.normalize(X_train)
      X_test = (X_test - self.mean) / self.std


    predictions = []
    for x in X_test:
      distances = self.distances(x, X_train)
      indices = np.argsort(distances)[:k]
      labels = y_train[indices]
      prediction = np.bincount(labels).argmax()
      predictions.append(prediction)

    return predictions
