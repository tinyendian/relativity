import numpy as np

# Use "new-style" classes for easier inheritance
__metaclass__ = type

class metric:
  """
  Base class for defining a metric as a symmetric bilinear form
  """
  def __init__(self):
    # This does not currently take symmetries into account
    self.name = "Null"
    self.matrix = np.zeros((4,4), dtype = np.float64)
    self.christoffel = np.zeros((4,4,4), dtype = np.float64)

  def __eq__(self, other):
    """
    Requires that operand is a metric with same name and
	numerically near-identical matrix elements and Christoffel
	symbols
    """
    result = isinstance(other, metric)
    result = result and (self.name == other.name)
    result = result and np.allclose(self.matrix, other.matrix)
    result = result and np.allclose(self.christoffel, other.christoffel)
    return result

  def __str__(self):
    return self.matrix.__str__() + "\nName: " + self.name

  def __ne__(self, other):
    return not self.__eq__(other)

  def getName(self):
    return self.name

  def getMatrix(self):
    return self.matrix

  def getChristoffel(self):
    return self.christoffel

  def updateCoords(self, coord):
    pass

  def scalarProduct(self, v, w):
    """
    Returns scalar product of two contravariant vectors,

    v^i * g_ij * w^j
    """
    return np.einsum('i,ij,j', v, self.matrix, w)

# -----------------------------------------------------------------------

class minkowski(metric):
  """
  Defines the Minkowski metric
  g_00 = 1
  g_11 = g_22 = g_33 = -1
  
  All Christoffel symbols vanish
  """
  def __init__(self):
    super(minkowski, self).__init__()
    self.name = 'Minkowski'
    self.matrix = np.diagflat(np.array([1,-1,-1,-1], dtype = np.float64))

# -----------------------------------------------------------------------

class schwarzschild(metric):
  """
  Defines the Schwarzschild metric in spherical coordinates and proper time
  of a distant observer
  """

  def __init__(self, rSchwarzschild, r, theta):
    """
    Set Schwarzschild radius in arbitrary units
    """
    assert(rSchwarzschild >= 0), "Schwarzschild radius cannot be negative"
    assert(r != rSchwarzschild), "Metric diverges at Schwarzschild radius"
    assert(theta >= 0 and theta <= np.pi), "Polar angle must be in range 0..pi"

    super(schwarzschild, self).__init__()
    self.name = 'Schwarzschild'
    self.rSchwarzschild = rSchwarzschild
    self.r = r
    self.theta = theta
    self.updateCoords([0, r, theta, 0])

  def updateCoords(self, coord):
    if isinstance(coord, np.ndarray):
      assert (coord.shape == (4,)), "Coordinate tuple must have 4 components"
    else:
      assert (len(coord) == 4), "Coordinate tuple must have 4 components"

    r = coord[1]
    theta = coord[2]

    self.matrix[0,0] = 1-self.rSchwarzschild/r
    self.matrix[1,1] = -1/self.matrix[0,0]
    self.matrix[2,2] = -r*r
    self.matrix[3,3] = -r*r*np.sin(theta)*np.sin(theta)

    # Time components
    self.christoffel[0,0,1] = self.christoffel[0,1,0] = 0.5*self.rSchwarzschild/(r*(r-self.rSchwarzschild))
	
    # Radius components
    self.christoffel[1,0,0] = 0.5*self.rSchwarzschild*(r-self.rSchwarzschild)/(r*r*r)
    self.christoffel[1,1,1] = -self.christoffel[0,0,1]
    self.christoffel[1,2,2] = -(r-self.rSchwarzschild)
    self.christoffel[1,3,3] = self.christoffel[1,2,2]*np.sin(theta)*np.sin(theta)

    # Azimuth components
    self.christoffel[2,1,2] = self.christoffel[2,2,1] = 1/r
    self.christoffel[2,3,3] = -np.sin(theta)*np.cos(theta)

    # Polar components
    self.christoffel[3,1,3] = self.christoffel[3,3,1] = 1/r
    if not np.isclose(np.tan(theta),0):
      self.christoffel[3,2,3] = self.christoffel[3,3,2] = 1/np.tan(theta)
