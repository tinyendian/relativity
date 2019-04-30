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
    self.integralCurve = None

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
                           y0, method = 'RK45', t_eval = times, dense_output = True)

    # Let user know if things went wrong, but keep output nonetheless
    if not result["success"]:
      print(result)

    # Store integration results - OdeSolution object, proper time, coords, velocities
    self.integralCurve = result["sol"]
    for i in range(len(result["t"])):
      self.curveparam.append(result["t"][i])
      coord = result["y"][0:4,i]
      self.coords.append(coord)
      # Use deep copy here so that we get the correct child class
      self.velocities.append(copy.deepcopy(self.velocity0))
      self.velocities[-1].vector = result["y"][4:8,i]
      # Each velocity vector has its own copy of the metric, evaluated at coord
      self.velocities[-1].metric.updateCoords(coord)

  def acceleration(self, properTime):
    """
    Computes acceleration along a worldline using the covariant derivative,
    
    D_v(v)^k = dv^k/dtau + v^i*v^j*gamma^k_ij
    
    where gamma are the Christoffel symbols.
    """
    assert (properTime >= 0), "Proper time must be >= 0"
    assert (self.integralCurve is not None), "Worldline must have at least two points"

    # Get proper time at the end of worldline
    totalTime = self.curveparam[-1]
    assert (properTime <= totalTime), "Proper time larger than end time of worldline"

    # Compute directional derivative along curve around given proper time using a
    # +- 5% stencil
    dtau = 0.05 * totalTime
    # Use two-sided stencil if possible
    if properTime >= dtau and properTime <= (totalTime-dtau):
      dvdtau = (self.integralCurve(properTime+dtau)[4:8]-self.integralCurve(properTime-dtau)[4:8])/(2*dtau)
    elif properTime > (totalTime-dtau):
      dvdtau = (self.integralCurve(properTime)[4:8]-self.integralCurve(properTime-dtau)[4:8])/dtau
    elif properTime < dtau:
      dvdtau = (self.integralCurve(properTime+dtau)[4:8]-self.integralCurve(properTime)[4:8])/dtau
    else:
      print ("ERROR: This branch should never be reached")
      return None

    # Get coordinates and velocities at given proper time
    coord = self.integralCurve(properTime)[0:4]
    vel = self.integralCurve(properTime)[4:8]

    # Create new four-vector with the same metric as the worldline
    result = fv.fourvector([0,0,0,0], self.velocity0.metric)
    result.metric.updateCoords(coord)

    # Compute basis correction for directional derivative using Christoffel symbols
    corr = np.einsum('i,j,kij', vel, vel, result.metric.getChristoffel())

    # Combine directional derivative and basis correction to obtain covariant derivative
    result.vector = dvdtau + corr

    return result
