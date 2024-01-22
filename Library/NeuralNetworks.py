import numpy as np
import UsefulFunctions as uf

class NeuralNetwork:

  def __init__(self):
    self.model = []
    self.loss_history = []
    self.mean = 0
    self.std = 1

  def xavier_init(self, input, output):
    variance = 2.0 / (input + output)
    return np.random.randn(input, output) * np.sqrt(variance)

  def relu(self, z):
    return np.maximum(0, np.minimum(z,1e15))

  def linear(self, z):
    return z

  def softmax(self, z):
    return np.exp(np.clip(z, -700, 700)) / (np.sum(np.exp(np.clip(z, -700, 700)), axis=1))

  function_map = {
    'relu': relu,
    'linear': linear,
    'softmax': softmax
  }

  def relu_derivative(self, z):
    return (z > 0) * 1

  def linear_derivative(self, z):
    return np.ones_like(z)

  derivative_map = {
    'relu': relu_derivative,
    'linear': linear_derivative,
  }

  def Categorical_Crossentropy(self, true, pred):
    pred = np.clip(pred, 1e-15, 1-(1e-15))
    return -np.sum(np.multiply(true, np.log(pred)), axis=1)

  def Crossentropy_gradient(self, true, pred):
    pred = np.clip(pred, 1e-15, 1-(1e-15))
    return np.subtract(pred,true)

  def dense(self, units, activation):
    """
    Initializes layer of a neural network

    Arguments:
    units (scalar) : number of units 
    activation (string) : activation function required in this layer
    
    Available activation functions: relu, linear, softmax(only for final layer)
    """
    return (units, activation)

  def Sequential_model(self, Input_size, layers):
    """
    Initializes a sequential neural network model

    Arguments:
    Input_size (tuple) : size of the training data
    layers : To be initialized using dense function

    """
    self.model = []
    m, n = Input_size

    for i in range(len(layers)):
      units, activation = layers[i]
      if i == 0:
        self.model.append([self.xavier_init(n, units), self.xavier_init(1, units), np.zeros((m, n)), activation])
      else:
        Input = self.model[i - 1][0].shape[1]
        self.model.append([self.xavier_init(Input, units), self.xavier_init(1, units), np.zeros((m, Input)), activation])

    pass

  def compute(self, layer):
    W, b, a_in, g = layer
    z = np.matrix(np.matmul(a_in, W) + b)
    a_out = self.function_map.get(g)(self, z)
    return a_out

  def fit(self, X, y, alpha, epochs, batch_size, normalize=True):
    """
    Fits the model using mini-batch gradient descent

    Arguments:
    X (matrix) : Dataset of m examples with n features
    y (vector) : Target values
    alpha (scalar) : Learning rate
    epochs (scalar) : number of epochs to be computed
    batch_size (scalar) : number of examples to be taken for mini-batch
    normalize (boolean) : indicates if data is needed to be normalized
    """
    if(normalize):
      X, self.mean, self.std = uf.normalize(X)
      
    self.loss_history = []
    for epoch in range(epochs):

      self.model[0][2] = X

      for i in range(len(self.model)-1):
        self.model[i+1][2] = self.compute(self.model[i])

      y_pred = self.compute(self.model[-1])
      loss = self.Categorical_Crossentropy(y, y_pred)
      print("Epoch:", epoch, "   Loss:", np.sum(loss), end='')
      self.loss_history.append(np.sum(loss))

      for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        end = min(end, X.shape[0])

        self.model[0][2] = X[start:end]
        y_batch = y[start:end]

        for i in range(len(self.model)-1):
          self.model[i+1][2] = self.compute(self.model[i])

        f = self.compute(self.model[-1])

        delta = self.Crossentropy_gradient(y_batch, f)
        W, b, x, g = self.model[-1]
        b = b.reshape(1,-1)
        W -= alpha*(np.matmul(x.T, delta)/x.shape[0])
        b -= alpha*(np.mean(delta, axis=0))
        delta = np.matmul(delta,W.T)

        for W, b, x, g in reversed(self.model[:-1]):
          z = np.matrix(np.matmul(x, W) + b)
          delta = np.multiply(delta, self.derivative_map.get(g)(self, z))
          b = b.reshape(1,-1)
          W -= alpha*(np.matmul(x.T, delta)/x.shape[0])
          b -= alpha*(np.mean(delta, axis=0))
          delta = np.matmul(delta,W.T)
      print('',end='\r')

    pass

  def predict(self, X):
    """
    Predicts values using learned parameters

    Arguments:
    X (matrix) : Data to be made predictions of

    Returns: 
    y (vector) : predicted values
    """
    X = (X - self.mean)/self.std

    self.model[0][2] = X
    for i in range(len(self.model)-1):
      self.model[i+1][2] = self.compute(self.model[i])
    y = self.compute(self.model[-1])

    y = np.argmax(y, axis=1)

    return y