import UsefulFunctions as uf
import numpy as np

class LinearRegression:

  def __init__(self):
    self.cost_history = []
    self.w = []
    self.b = 0
    self.mean = 0
    self.std = 1

  def cost(self, X, y):
    """
    Calculates value of cost function

    Arguments: 
    X (matrix) : Dataset of m examples with n features
    y (vector) : Target values

    Returns:
    cost (scalar) : Cost of the iteration
    """
    f = np.matmul(X, self.w) + self.b
    cost = np.sum((f-y)**2)

    return cost

  def gradient(self, X, y):
    """
    Computes gradient of cost wrt w and b

    Arguments: 
    X (matrix) : Dataset of m examples with n features
    y (vector) : Target values

    Returns:
    dj_dw (vector) : Gradient wrt w
    dj_db (scalar) : Gradient wrt b
    """
    z = np.matmul(X, self.w) + self.b
    dj_dw = np.matmul(z-y, X) / X.shape[0]
    dj_db = np.mean(z-y)

    return dj_dw, dj_db

  def fit(self, X, y, alpha, epochs, normalize=True):
    """
    Fits the model wrt data

    Arguments:
    X (matrix) : Dataset of m examples with n features
    y (vector) : Target values
    alpha (scalar) : Learning rate
    epochs (scalar) : number of epochs to be computed
    normalize (boolean) : indicates if data is needed to be normalized
    """
    if(normalize):
      X, self.mean, self.std = uf.normalize(X)

    self.cost_history = []
    self.w = np.zeros(X.shape[1])
    self.b = 0

    for i in range(epochs):
      dj_dw, dj_db = self.gradient(X, y, self.w, self.b)
      self.w -= alpha*dj_dw
      self.b -= alpha*dj_db
      cost = self.cost(X, y, self.w, self.b)
      self.cost_history.append(cost)

    return self

  def predict(self, X):
    """
    Predicts values using learned parameters

    Arguments:
    X (matrix) : Data to be made predictions of

    Returns: 
    y (vector) : predicted values
    """
    X = (X - self.mean) / self.std
    y = np.matmul(X, self.w) + self.b

    return y
