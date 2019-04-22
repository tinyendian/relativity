import numpy as np
import fourvector as fv
import metric as mt

def test_fourvector():

  metric = mt.minkowski()

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
  assert (np.isclose(vtime.innerProduct(vspace), 0)), "Expeced zero result"
