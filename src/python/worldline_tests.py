import numpy as np
import worldline as wl
import fourvector as fv
import metric as mt

def test_geodesic():

  # Resting observer in flat Minkowski spacetime - only time coordinate should change
  vel0 = fv.observer([1,0,0,0], mt.minkowski())
  path = wl.worldline([0,0,0,0], vel0)
  path.geodesic(10)
  for i in range(len(path.curveparam)):
    assert(np.allclose(path.coords[i],[path.curveparam[i],0,0,0])), "Only time coordinate should change"
    assert(np.allclose(path.velocities[i].vector,[1,0,0,0])), "Observer should not accelerate"

  # Photon in flat Minkowski spacetime - only time and on space coordinate should change
  vel0 = fv.photon([1,0,1,0], mt.minkowski(), 1)
  path = wl.worldline([0,0,0,0], vel0)
  path.geodesic(10)
  for i in range(len(path.curveparam)):
    assert(np.allclose(path.coords[i],[path.curveparam[i],0,path.curveparam[i],0])), "Time and one space coordinate should change"
    assert(np.allclose(path.velocities[i].vector,[1,0,1,0])), "Light should not accelerate"
