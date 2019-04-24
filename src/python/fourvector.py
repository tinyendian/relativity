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
      assert(np.allclose(self.metric.getMatrix(), second.metric.getMatrix())), "Four-vectors must have same metric"
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
    assert (self.metric.getName() == 'Minkowski'), "Lorentz transformations require Minkowski metric"
    rot = tf.lorentzRotation(axis, angle)
    self.vector = rot.transformContraVector(self.vector)
    self.metric.matrix = rot.transformCoMatrix(self.metric.matrix)

  def lorentzBoost(self, axis, beta):
    assert (self.metric.getName() == 'Minkowski'), "Lorentz transformations require Minkowski metric"
    boost = tf.lorentzBoost(axis, beta)
    self.vector = boost.transformContraVector(self.vector)
    self.metric.matrix = boost.transformCoMatrix(self.metric.matrix)

class observer(fourvector):
  """
  Specialisation of the fourvector class to define observers.
  Observers are always time-like and normalised,
  
  g(x,x) = 1
  """

  def __init__(self, data, metric):
    super(observer, self).__init__(data, metric)
    assert (self.isTimeLike()), "Observers must be time-like"
    assert (np.isclose(self.norm(), 1)), "Observers must be time-like"

  def relativeVelocity(self, x):
    """
	Return space-like four-vector with relative velocity
	of this observer y as seen by another observer x,
	
	v = y/g(x,y) - x
	
	Note that v is in the orthogonal space of x,
	
	g(v,x) = 0
	"""
    assert(np.allclose(self.metric.getMatrix(), x.metric.getMatrix())), "Four-vectors must have same metric"
    return fourvector(self.vector/self.innerProduct(x) - x.vector, self.metric)

  def speed(self, x):
    """
	Return speed (in units of speed of light) of this observer
	as seen by another observer x
	"""
    assert(np.allclose(self.metric.getMatrix(), x.metric.getMatrix())), "Four-vectors must have same metric"
    return np.sqrt(-self.relativeVelocity(x).innerProduct())

class particle(observer):
  """
  Specialisation of the observer class to define particles
  with restmass.
  """
  def __init__(self, data, metric, restmass):
    super(particle, self).__init__(data, metric)
    assert (restmass > 0), "Mass must be positive"
    self.restmass = restmass

  def energy(self, x):
    assert(np.allclose(self.metric.getMatrix(), x.metric.getMatrix())), "Four-vectors must have same metric"
    return self.restmass*self.innerProduct(x)

class photon(fourvector):
  """
  Specialisation of the fourvector class to define photons
  with given frequency. Photons are always light-like
  (obviously...),
  
  g(x,x) = 0
  """

  def __init__(self, data, metric, frequency):
    super(photon, self).__init__(data, metric)
    assert (self.isLightLike()), "Photons must be light-like"
    self.frequency = frequency
