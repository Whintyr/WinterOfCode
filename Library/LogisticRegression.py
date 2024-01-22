import UsefulFunctions as uf
import numpy as np

class LogisticRegression:
  def __init__(self):
    self.w = 0
    self.std = 1
    self.mean = 0
    self.normalize = True
    self.bias = True

  def sigmoid(self, x):
    # Computes the sigmoid of the given input
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

  def fit(self, X, y, alpha, epochs, normalize=True, bias=True):
    """
    Fits the model wrt data

    Arguments:
    X (matrix) : Dataset of m examples with n features
    y (vector) : Target values
    alpha (scalar) : Learning rate
    epochs (scalar) : number of epochs to be computed
    normalize (boolean) : indicates if data is needed to be normalized
    bias (boolean) : indicates if bias is needed to be added to data
    """

    self.normalize = normalize
    self.bias = bias

    if(self.normalize):
      X, self.mean, self.std = uf.normalize(X)

    if(self.bias):
      X = uf.add_bias(X)

    self.w = np.zeros((X.shape[1], y.shape[1]))

    for epoch in range(epochs):
      z = np.matmul(X, self.w)
      f = self.sigmoid(z)
      dw = np.matmul(X.T, (f - y)) / y.shape[0]
      self.w -= alpha * dw
    pass

  def predict(self, X):
    """
    Predicts values using learned parameters

    Arguments:
    X (matrix) : Data to be made predictions of

    Returns: 
    y (vector) : predicted values
    """
    if(self.normalize):
      X = (X - self.mean)/self.std

    if(self.bias):
      X = uf.add_bias(X)

    z = np.matmul(X, self.w)
    y = self.sigmoid(z)

    return y