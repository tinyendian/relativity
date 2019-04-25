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

  #
  # Operator definitions
  #

  # Only implement essential operators and derive the others from these
  def __add__(self, other):
    assert(isinstance(other, fourvector)), "Argument must be fourvector type"
    assert(self.metric == other.metric), "Four-vectors must have same metric"
    return fourvector(self.vector+other.vector, self.metric)

  def __mul__(self, other):
    assert(isinstance(other, (int,float))), "Argument must be int or float type"
    return fourvector(self.vector*other, self.metric)

  def __eq__(self, other):
    """
    Requires that operand is a fourvector with identical components and
	numerically near-identical metric
    """
    result = isinstance(other, fourvector)
    result = result and np.array_equal(self.vector, other.vector)
    result = result and self.metric == other.metric
    return result

  def __sub__(self, other):
    return self.__add__(other*(-1.))

  def __rmul__(self, other):
    return self.__mul__(other)

  def __truediv__(self, other):
    return self.__mul__(1./other)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __getitem__(self, index):
    return self.vector[index]

  def __str__(self):
    return self.vector.__str__() + " Metric: " + self.metric.name

  #
  # Methods
  #

  def innerProduct(self, other = None):
    """
    Return inner product of vector with itself,
    g(v,v)
    or with another vector,
    g(v,w)
    """
    if other is None:
      return self.metric.scalarProduct(self.vector, self.vector)
    else:
      assert(isinstance(other, fourvector)), "Argument must be fourvector type"
      assert(self.metric == other.metric), "Four-vectors must have same metric"
      return self.metric.scalarProduct(self.vector, other.vector)

  def norm(self):
    return np.sqrt(np.abs(self.innerProduct()))

  def isTimeLike(self):
    return self.innerProduct() > self.tolerance

  def isLightLike(self):
    return np.abs(self.innerProduct()) < self.tolerance

  def isSpaceLike(self):
    return self.innerProduct() < -self.tolerance

  def lorentzRotate(self, axis, angle):
    """
    Spatial rotation around given axis with given angle
    """
    assert (self.metric.getName() == 'Minkowski'), "Lorentz transformations require Minkowski metric"
    rot = tf.lorentzRotation(axis, angle)
    self.vector = rot.transformContraVector(self.vector)

  def lorentzBoost(self, axis, beta):
    """
    Boost into frame that moves along given axis at speed beta = v/c
    """
    assert (self.metric.getName() == 'Minkowski'), "Lorentz transformations require Minkowski metric"
    boost = tf.lorentzBoost(axis, beta)
    self.vector = boost.transformContraVector(self.vector)

class observer(fourvector):
  """
  Specialisation of the fourvector class to define observers as normalised
  four-velocities. Observers are always time-like with
  
  g(x,x) = 1
  
  in natural units (c=1).
  """

  def __init__(self, data, metric):
    super(observer, self).__init__(data, metric)
    assert (self.isTimeLike()), "Observers must be time-like"
    assert (np.isclose(self.norm(), 1)), "Observers must be time-like"

  def relativeVelocity(self, other):
    """
	Return space-like four-vector with relative velocity
	of this observer y as seen by another observer x,
	
	v = y/g(x,y) - x
	
	Note that v is in the orthogonal space of x,
	
	g(v,x) = 0
	"""
    assert(self.metric == other.metric), "Four-vectors must have same metric"
    return self/self.innerProduct(other) - other

  def speed(self, other):
    """
	Return speed (in units of speed of light) of this observer
	as seen by another observer x
	"""
    assert(isinstance(other, fourvector)), "Argument must be fourvector type"
    assert(self.metric == other.metric), "Four-vectors must have same metric"
    return np.sqrt(-self.relativeVelocity(other).innerProduct())

class particle(observer):
  """
  Specialisation of the observer class to define particles
  with restmass.
  """
  def __init__(self, data, metric, restmass):
    super(particle, self).__init__(data, metric)
    assert (restmass > 0), "Mass must be positive"
    self.restmass = restmass

  def energy(self, other):
    assert(isinstance(other, fourvector)), "Argument must be fourvector type"
    assert(self.metric == other.metric), "Four-vectors must have same metric"
    return self.restmass*self.innerProduct(other)

class photon(fourvector):
  """
  Specialisation of the fourvector class to define photons
  with given energy. Photons are always light-like
  (obviously...),
  
  g(x,x) = 0
  """

  def __init__(self, data, metric, energy):
    super(photon, self).__init__(data, metric)
    assert (self.isLightLike()), "Photons must be light-like"
    self.energy = energy
