import numpy as np

# Use "new-style" classes for easier inheritance
__metaclass__ = type

class transformation:
  def __init__(self):
    self.matrix = np.zeros((4,4), dtype = np.float64)
    self.invmatrix = np.zeros((4,4), dtype = np.float64)

  def getMatrix(self):
    return self.matrix

  def getInvMatrix(self):
    return self.invmatrix

  def transformContraVector(self, v):
    """
    Transforms column vectors (1-times contravariant tensors).

    x'^i = T^i_j * x^j  <=>  x' = Tx
    
    Matrix index convention is row-column
    """
    # See transformation rule for x^j above    
    return np.einsum('ij,j', self.matrix, v)

  def transformCoMatrix(self, m):
    """
    Transform matrices representing bilinear forms (2-times covariant tensors).
    
    We start with using the transformation T for contravariant vectors,

    g(x',x') = x'^i * g'_i_j * x'^j
             = T^i_k * x^k * g'_i_j * T^j_l * x^l
             = x^k * (T^i_k * g'_i_j * T^j_l) * x^l
    
    We thus get
    
    g_k_l = T^i_k * g'_i_j * T^j_l

    Applying the inverse matrices on both sides,
    
    T^i_k * (T^-1)^k_j = delta^i_j
    
    where delta is the Kronecker symbol, we get the transformation rule
    
    g'_i_j = (T^-1)^k_i * g_k_l * (T^-1)^l_j
    
    The bilinear form thus transforms with the inverse matrix T^-1, hence
    its covariant transformation behaviour. To write this as a matrix product,
    we need to use the matrix transpose (the order of indices is important!),
    
    (T^-1)^k_i = (T^-1^t)_i^k
    
    which results in

    g' = T^-1^t g T^-1
    """
    # See transformation rule for g_k_l above    
    return np.einsum('ki,kl,lj', self.invmatrix, m, self.invmatrix)

# -----------------------------------------------------------------------

class lorentzRotation(transformation):
  """
  Rotations of contravariant vectors in counter-clockwise direction (for angle > 0)
  in a right-handed coordinate system; the inverse matrix is used for covariant
  vectors
  """
  def __init__(self, axis, angle):
    assert (isinstance(axis, int)), "Axis parameter must be an integer number"
    assert (axis > 0 and axis < 4), "Axis must be in range 1..3"
    assert (isinstance(angle, (int, float))), "Angle parameter must be a number"
    self.axis = axis
    self.angle = angle

    super(lorentzRotation, self).__init__()

    self.matrix[0,0] = 1
    if axis == 1:
      self.matrix[1,1] = 1
      self.matrix[2,2] = self.matrix[3,3] = np.cos(self.angle)
      self.matrix[2,3] = -np.sin(self.angle)
      self.matrix[3,2] = np.sin(self.angle)
    elif axis == 2:
      self.matrix[2,2] = 1
      self.matrix[1,1] = self.matrix[3,3] = np.cos(self.angle)
      self.matrix[1,3] = np.sin(self.angle)
      self.matrix[3,1] = -np.sin(self.angle)
    elif axis == 3:
      self.matrix[3,3] = 1
      self.matrix[1,1] = self.matrix[2,2] = np.cos(self.angle)
      self.matrix[1,2] = -np.sin(self.angle)
      self.matrix[2,1] = np.sin(self.angle)

    # Rotation matrices are orthogonal
    self.invmatrix = np.transpose(self.matrix)

# -----------------------------------------------------------------------

class lorentzBoost(transformation):
  """
  Boosts of contravariant vectors, the inverse matrix is used for covariant
  vectors
  """
  def __init__(self, axis, beta):
    assert (isinstance(axis, int)), "Axis parameter must be an integer number"
    assert (axis > 0 and axis < 4), "Axis must be in range 1..3"
    assert (isinstance(beta, (int, float))), "Angle parameter must be a number"
    assert (np.abs(beta) < 1), "Beta must be > -1 and < 1"
    self.axis = axis
    self.beta = beta
    self.rapidity = np.arctanh(self.beta)

    super(lorentzBoost, self).__init__()

    if axis == 1:
      self.matrix[0,0] = np.cosh(self.rapidity)
      self.matrix[0,1] = -np.sinh(self.rapidity)
      self.matrix[1,0] = -np.sinh(self.rapidity)
      self.matrix[1,1] = np.cosh(self.rapidity)
      self.matrix[2,2] = 1
      self.matrix[3,3] = 1
    elif axis == 2:
      self.matrix[0,0] = np.cosh(self.rapidity)
      self.matrix[0,2] = -np.sinh(self.rapidity)
      self.matrix[1,1] = 1
      self.matrix[2,0] = -np.sinh(self.rapidity)
      self.matrix[2,2] = np.cosh(self.rapidity)
      self.matrix[3,3] = 1
    elif axis == 3:
      self.matrix[0,0] = np.cosh(self.rapidity)
      self.matrix[0,3] = -np.sinh(self.rapidity)
      self.matrix[1,1] = 1
      self.matrix[2,2] = 1
      self.matrix[3,0] = -np.sinh(self.rapidity)
      self.matrix[3,3] = np.cosh(self.rapidity)

    # Off-diagonal elements of inverse matrix change sign
    self.invmatrix = -self.matrix
    self.invmatrix[0,0] *= -1
    self.invmatrix[1,1] *= -1
    self.invmatrix[2,2] *= -1
    self.invmatrix[3,3] *= -1
