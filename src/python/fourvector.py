import numpy as np
import transformation as tf

class fourvector:
  """
  Defines a contravariant four-vector for relativistic computations.
  Each four-vector carries a reference to a metric which is used for
  computing inner product, norm, etc.
  """

  # Threshold for an inner product to be considered "light-like",
  # |g(v,v)| < tol
  tolerance = 1.0e-10

  def __init__(self, data, metric):
    assert (len(data) == 4), "Expect input vector with 4 components"
    self.vector = np.array(data)
    self.metric = metric

  def innerProduct(self, second = None):
    """
    Return inner product of vector with itself,
    g(v,v)
    or with another vector,
    g(v,w)
    """
    if second is None:
      return self.metric.scalarProduct(self.vector, self.vector)
    else:
      return self.metric.scalarProduct(self.vector, second.vector)

  def norm(self):
    return np.sqrt(np.abs(self.innerProduct()))

  def isTimeLike(self):
    return self.innerProduct() > self.tolerance

  def isLightLike(self):
    return np.abs(self.innerProduct()) < self.tolerance

  def isSpaceLike(self):
    return self.innerProduct() < -self.tolerance

  def lorentzRotate(self, axis, angle):
    rot = tf.lorentzRotation(axis, angle)
    self.vector = rot.transformContraVector(self.vector)
    self.metric.matrix = rot.transformCoMatrix(self.metric.matrix)

  def lorentzBoost(self, axis, beta):
    boost = tf.lorentzBoost(axis, beta)
    self.vector = boost.transformContraVector(self.vector)
    self.metric.matrix = boost.transformCoMatrix(self.metric.matrix)
