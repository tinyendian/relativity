import numpy as np

# Use "new-style" classes for easier inheritance
__metaclass__ = type

class metric:
  """
  Base class for defining a metric as a symmetric bilinear form
  """
  def __init__(self):
    # This does not currently take symmetries into account
    self.name = None
    self.matrix = np.zeros((4,4), dtype = np.float64)
    self.christoffel = np.zeros((4,4,4), dtype = np.float64)

  def getName(self):
    return self.name

  def getMatrix(self):
    return self.matrix

  def getChristoffel(self):
    return self.christoffel

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

    self.matrix[0,0] = 1-self.rSchwarzschild/r
    self.matrix[1,1] = -1/self.matrix[0,0]
    self.matrix[2,2] = -r*r
    self.matrix[3,3] = -r*r*np.sin(theta)*np.sin(theta)
