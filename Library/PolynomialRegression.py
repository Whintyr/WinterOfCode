import UsefulFunctions as uf
import numpy as np

class PolynomialRegression:

  def __init__(self):
    self.cost_history = []
    self.w = []
    self.b = 0
    self.mean = 0
    self.std = 0
    self.degree = 0

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

  def fit(self, X, y, degree, alpha, epochs, normalize=True):
    """
    Fits the model wrt data

    Arguments:
    X (matrix) : Dataset of m examples with n features
    y (vector) : Target values
    degree (scalar) : degree of polynomial required
    alpha (scalar) : Learning rate
    epochs (scalar) : number of epochs to be computed
    normalize (boolean) : indicates if data is needed to be normalized
    """
    self.degree = degree
    X = self.poly_features(self.degree, X)

    if(normalize):
      X, self.mean, self.std = uf.normalize(X)

    self.cost_history = []
    self.w = np.zeros(X.shape[1])
    self.b = 0

    for i in range(epochs):
      dj_dw, dj_db = self.linear_gradient(X, y)
      self.w -= alpha*dj_dw
      self.b -= alpha*dj_db
      cost = self.linear_cost(X, y)
      self.cost_history.append(cost)
    
    pass

  def predict(self, X, normalize=True):
    """
    Predicts values using learned parameters

    Arguments:
    X (matrix) : Data to be made predictions of

    Returns: 
    y (vector) : predicted values
    """
    X = self.poly_features(self.degree, X)

    if(normalize):
      X = (X - self.mean) / self.std

    y = np.matmul(X, self.w) + self.b

    return y

  def poly_features(self, degree, X):
    """
    Polynomializes data to the given degree

    Arguments: 
    degree (scalar) : degree of polynomial
    X (matrix) : Dataset of m examples and n features

    Returns:
    X_poly (matrix) : Dataset with polynomialzed features
    """
    features = [i for i in range(X.shape[1])]
    combinations = self.generate_feature_combinations(features, degree)

    X_poly = np.ones((X.shape[0], len(combinations)))

    for i, combination in enumerate(combinations):
      for j, power in enumerate(combination):
        X_poly[:, i] *= X[:, j]**power

    X_poly = X_poly[:, 1:]

    return X_poly

  def generate_feature_combinations(self, features, degree, current_combination=[], current_degree=0, index=0):
    """
    Recursive method which generates all combinations of features

    Arguments:
    features (vector) : vector with all features
    degree (scalar) : degree of polynomial
    current_combination (vector) : stores the ongoing combination, default=[]
    current_degree (scalar) : stores the total degree of ongoing combination, default=[]
    current_index (scalar) : stores the index of feature being traversed, default=0

    Returns:
    combinations (matrix) : contains all possible combinations of features
    """
    if current_degree > degree:
      return []

    if index == len(features):
      return [current_combination]

    combinations = []

    for i in range(degree - current_degree + 1):
      new_combination = current_combination + [i]
      combinations.extend(self.generate_feature_combinations(features, degree, new_combination, current_degree + i, index + 1))

    return combinations
