import numpy as np
import scipy.integrate as spi
import fourvector as fv
import copy

def geodesicRHS(s, x, metric):
  """
  Right-hand side of geodesic equations as a system of first-order
  ODEs in coordinate location x and velocity v as functions of curve
  parameter s,

  x'^i = v^i
  v'^i = -gamma^i_jk * v^j * v^k

  where gamma^i_jk are the Christoffel symbols for given metric.
  """
  metric.updateCoords(x[0:4])
  result = np.zeros(8, dtype = np.float64)
  result[0:4] = x[4:8]
  result[4:8] = -np.einsum('ijk,j,k',metric.getChristoffel(),x[4:8],x[4:8])
  return result

class worldline:

  def __init__(self, coord0, velocity0):
    if isinstance(coord0, np.ndarray):
      assert (coord0.shape == (4,)), "Start coordinate tuple must have 4 components"
      self.coord0 = coord0.astype(np.float64)
    else:
      assert (len(coord0) == 4), "Start coordinate tuple must have 4 components"
      self.coord0 = np.array(coord0, dtype = np.float64)
    assert(isinstance(velocity0, fv.fourvector)), "Start velocity must be fourvector type"
    self.velocity0 = velocity0

    # Use plain Python list to store wordlines for convenience, may need to replace this with
    # NumPy arrays at some point for better performance
    self.curveparam = []
    self.coords = []
    self.velocities = []

  def geodesic(self, properTime, nSteps = None):
    """
    Evolve a coordinate tuple and velocity fourvector along a geodesic using the
    geodesic equations,

    x''^i + gamma^i_jk * x'^j * x'^k = 0

    where x' and x'' are first and second derivatives with respect to proper time.
    Argument properTime sets the integration limit, nSteps the number of integration
    steps that will be stored.
    """    
    assert (properTime >= 0), "Proper time must be >= 0"

    # Reset any previous computation
    self.curveparam.clear()
    self.coords.clear()
    self.velocities.clear()

    if properTime == 0:
      self.curveparam.append(0)
      self.coords.append(self.coord0)
      self.velocities.append(self.velocity0)

    if nSteps is not None:
      assert (nSteps > 0), "nSteps must be 1 or larger"
      times = np.linspace(0, properTime, nSteps)
    else:
      times = None

    # Set up vector with initial values
    y0 = np.concatenate((self.coord0, self.velocity0.vector))

    result = spi.solve_ivp(lambda t,y: geodesicRHS(t,y,self.velocity0.metric), [0, properTime],
                           y0, method = 'RK45', t_eval = times)

    # Let user know if things went wrong, but keep output nonetheless
    if not result["success"]:
      print(result)

    # Store integration results as lists - proper time, coordinate tuples, velocity vectors
    for i in range(len(result["t"])):
      self.curveparam.append(result["t"][i])
      coord = result["y"][0:4,i]
      self.coords.append(coord)
      # Use deep copy here so that we get the correct child class
      self.velocities.append(copy.deepcopy(self.velocity0))
      self.velocities[-1].vector = result["y"][4:8,i]
      # Each velocity vector has its own copy of the metric, evaluated at coord
      self.velocities[-1].metric.updateCoords(coord)
