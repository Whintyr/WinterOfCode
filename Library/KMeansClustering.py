import UsefulFunctions as uf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers

class KMeansClustering:
  def __init__(self):
    self.labels = []
    self.centres = []
    self.data = []
    self.mean = 0
    self.std = 1

  def distances(self, x1, X2):
    # Returns the distances of one point (x1[vector]) wrt multiple points (X2[matrix])
    return np.sum(np.abs(X2 - x1), axis=1)

  def clustering(self, data, k, epochs, normalize=True):
    """
    Implements KMeans Clustering algorithm for the given data

    Arguments:
    data (matrix) : Dataset with examples and its features
    k (scalar) : number of clusters
    epochs (scalar) : number of epochs
    normalize (boolean) : indicates if data is needed to be normalized
    """
    if(normalize):
      self.data, self.mean, self.std = uf.normalize(data)
    else:
      self.data = data
    self.centres = self.data[np.random.randint(0, self.data.shape[0], k)]
    self.labels = np.zeros(self.data.shape[0])
    
    for epoch in range(epochs):
      for i,x in enumerate(self.data):
        self.labels[i] = np.argmin(self.distances(x, self.centres))
      
      for i in range(len(self.centres)):
        cluster_points = self.data[np.where(self.labels==i)[0]]
        self.centres[i] = np.mean(cluster_points, axis=0)

    pass
  
  def plot(self, axes):
    """
    Plots a scatter plot of clusters wrt 2 features

    Arguments: 
    axes (tuple) : two features for which scatter plot is to be plotted
    """
    data = (self.data + self.mean)*self.std
    centres = (self.centres + self.mean)*self.std
    markers = list(mmarkers.MarkerStyle.markers.keys())
    for i in range(len(centres)):
      cluster_points = data[np.where(self.labels==i)[0]]
      plt.plot(cluster_points[:, axes[0]], cluster_points[:, axes[1]], markers[i+2], label=f"Cluster: {i}")

    plt.scatter(centres[:, axes[0]], centres[:, axes[1]], marker='X', color='black', label='Centroids')
    plt.legend()
    plt.show()
    pass
