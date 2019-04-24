import numpy as np
import fourvector as fv
import metric as mt

def test_fourvector():

  metric = mt.minkowski()

  # Check inner products and vector types
  particle = fv.fourvector([1,0,0,0], metric)
  assert (np.isclose(particle.innerProduct(), 1)), "Inner product not time-like"
  assert (particle.isTimeLike()), "Expected time-like vector"
  assert (not particle.isLightLike()), "Expected time-like vector"
  assert (not particle.isSpaceLike()), "Expected time-like vector"

  particle.lorentzRotate(1,0.321)
  assert (particle.isTimeLike()), "Expected time-like vector"
  particle.lorentzBoost(2,0.734)
  assert (particle.isTimeLike()), "Expected time-like vector"

  photon = fv.fourvector([1,1,0,0], metric)
  assert (np.isclose(photon.innerProduct(), 0)), "Inner product not light-like"
  assert (not photon.isTimeLike()), "Expected light-like vector"
  assert (photon.isLightLike()), "Expected light-like vector"
  assert (not photon.isSpaceLike()), "Expected light-like vector"

  photon.lorentzRotate(1,0.321)
  assert (photon.isLightLike()), "Expected light-like vector"
  photon.lorentzBoost(2,0.734)
  assert (photon.isLightLike()), "Expected light-like vector"

  space = fv.fourvector([0,1,0,0], metric)
  assert (np.isclose(space.innerProduct(), -1)), "Inner product not space-like"
  assert (not space.isTimeLike()), "Expected space-like vector"
  assert (not space.isLightLike()), "Expected space-like vector"
  assert (space.isSpaceLike()), "Expected space-like vector"

  space.lorentzRotate(1,0.321)
  assert (space.isSpaceLike()), "Expected space-like vector"
  space.lorentzBoost(2,0.734)
  assert (space.isSpaceLike()), "Expected space-like vector"

  vtime = fv.fourvector([1,0,0,0], metric)
  vspace = fv.fourvector([0,0,0,1], metric)
  assert (np.isclose(vtime.innerProduct(vspace), 0)), "Expected zero result"

def test_observer():

  metric = mt.minkowski()

  # Observer 1 in rest frame
  obs1 = fv.observer([1,0,0,0], metric)

  beta = 0.269
  for axis in range(1,4):

    # Observer 2 will be boosted into constant motion
    obs2 = fv.observer([1,0,0,0], metric)
    obs2.lorentzBoost(axis, beta)

    # Determine relative velocity of obs2 as seen by observer obs1
    relV = obs2.relativeVelocity(obs1)

    if axis == 1:
      assert (np.allclose(relV.vector,[0,-beta,0,0])), "Expected space-like vector with x1=-beta"
    elif axis == 2:
      # Beta < 0 in this case
      assert (np.allclose(relV.vector,[0,0,-beta,0])), "Expected space-like vector with x2=beta"
    elif axis == 3:
      assert (np.allclose(relV.vector,[0,0,0,-beta])), "Expected space-like vector with x3=-beta"

    # Relative velocity must be in orthogonal space of obs1
    assert (np.isclose(obs1.innerProduct(relV),0)), "Expected orthogonal relative velocity vector"

    # Speed is always a positive value
    assert (np.isclose(obs2.speed(obs1), np.abs(beta))), "Expected speed beta"

    # Swap boost direction for next axis
    beta *= -1

def test_particle():

  metric = mt.minkowski()

  # Moving particle with rest mass "mass"
  mass = 1.2345
  particle = fv.particle([1,0,0,0], metric, mass)
  beta = -0.987
  gamma = 1.0/np.sqrt(1-beta*beta)
  particle.lorentzBoost(2,beta)
  # Observer in rest frame
  observer = fv.observer([1,0,0,0], metric)
  assert (np.isclose(particle.energy(observer), gamma*mass)), "Expected energy gamma*mass"
