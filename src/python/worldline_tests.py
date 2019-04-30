import numpy as np
import worldline as wl
import fourvector as fv
import metric as mt

def test_geodesic():

  #
  # Flat spacetime
  #

  # Resting observer at origin in flat Minkowski spacetime - only time coordinate should change
  vel0 = fv.observer([1,0,0,0], mt.minkowski())
  path = wl.worldline([0,0,0,0], vel0)
  path.geodesic(10)
  for i in range(len(path.curveparam)):
    tau = path.curveparam[i]
    assert (np.allclose(path.coords[i], (tau*vel0).vector)), "Only time coordinate should change"
    assert (np.allclose(path.velocities[i].vector, vel0.vector)), "Four-velocity should be constant"
    assert (np.allclose(path.acceleration(tau).vector, np.zeros(4))), "Four-acceleration should vanish"

  # Particle in flat Minkowski spacetime - time and space coordinates should change
  # Mass is irrelevant here
  mass = 1
  vel0 = fv.particle([1,0,0,0], mt.minkowski(), mass)
  # Boost and rotate coordinate system, for the fun of it...
  beta = 0.9
  angle = np.pi/3
  vel0.lorentzBoost(3, beta)
  vel0.lorentzRotate(2, angle)
  vel0.lorentzRotate(3, -angle)
  path = wl.worldline([0,0,0,0], vel0)
  path.geodesic(10)
  # Define resting observer - same at every location of moving particle
  obs = fv.observer([1,0,0,0], mt.minkowski())
  for i in range(len(path.curveparam)):
    assert (np.allclose(path.coords[i], (path.curveparam[i]*vel0).vector)), "Time and space coordinates should change linearly with observer's proper time"
    assert (np.allclose(path.velocities[i].vector, vel0.vector)), "Four-velocity should be constant"
    # Measure particle speed in observer's frame at every location
    assert (np.isclose(path.velocities[i].speed(obs), beta)), "Resting observer should measure same speed"
    assert (np.allclose(path.acceleration(tau).vector, np.zeros(4))), "Four-acceleration should vanish"

  # Photon in flat Minkowski spacetime - time and space coordinates should change
  # Energy is irrelevant here
  energy = 1
  vel0 = fv.photon([1,0,1,0], mt.minkowski(), energy)
  path = wl.worldline([0,0,0,0], vel0)
  path.geodesic(10)
  for i in range(len(path.curveparam)):
    assert (np.allclose(path.coords[i], (path.curveparam[i]*vel0).vector)), "Time and one space coordinate should change"
    assert (np.allclose(path.velocities[i].vector, vel0.vector)), "Light should not accelerate"
    assert (np.allclose(path.acceleration(tau).vector, np.zeros(4))), "Four-acceleration should vanish"

  #
  # Curved (Schwarzschild) spacetime
  #

  # Particle in stable orbit around equator at r > 1.5*rs
  rs = 1.0
  t0 = 0.0
  r0 = 50*rs
  theta0 = 0.5*np.pi
  phi0 = 0.0
  # Mass is irrelevant here
  mass = 1
  # Compute initial four-velocity from geodesic equation for circular orbit
  # r'' = (r-3/2*rs)*(phi')**2 - rs/(2r**2) = 0
  # Solving this expression for orbital coordinate velocity phi' results in
  # phi' = sqrt( (rs/2) / (r**2 * (r-3/2*rs)) )
  vphi0 = np.sqrt(rs/(2*r0*r0*(r0-3*rs/2)))
  # Inserting the above expression into constraint g(x,x) = 1 for a particle,
  # (1-rs/r)*(t')**2 - r**2 * (phi')**2 = 1
  # results in
  # t' = sqrt( (1+(r*phi')**2)/(1-rs/r) )
  vt0 = np.sqrt((1+r0*r0*vphi0*vphi0)/(1-rs/r0))
  vel0 = fv.particle([vt0,0,0,vphi0], mt.schwarzschild(rs,r0,theta0), mass)
  path = wl.worldline([t0,r0,theta0,phi0], vel0)
  path.geodesic(100)
  # Define resting observer at each orbital location, using orbital symmetry
  # This observer is NOT on a geodesic!
  vtobs = 1.0/np.sqrt(1-rs/r0)
  obs = fv.observer([vtobs,0,0,0], mt.schwarzschild(rs,r0,theta0))
  for i in range(len(path.curveparam)):
    tau = path.curveparam[i]
    coord = path.coords[i]
    # Constant time coordinate velocity: t(tau) = tau*t'
    assert (np.isclose(coord[0], tau*vt0)), "Expected constant time coordinate velocity"
    assert (np.isclose(coord[1], r0)), "Expected constant radius coordinate"
    assert (np.isclose(coord[2], theta0)), "Expected constant polar angle coordinate"
    # Constant orbital coordinate velocity: phi(tau) = tau*phi'
    assert (np.isclose(coord[3], tau*vphi0)), "Expected constant orbital coordinate velocity"

    vel = path.velocities[i]
    assert (np.isclose(vel[0], vt0)), "Expected no acceleration of time"
    assert (np.isclose(vel[1], 0.0)), "Expected no radial motion"
    assert (np.isclose(vel[2], 0.0)), "Expected no polar angle motion"
    assert (np.isclose(vel[3], vphi0)), "Expected no orbital acceleration"

    assert (np.allclose(path.acceleration(tau).vector, np.zeros(4))), "Four-acceleration should vanish"

    # The resting observers must all measure the same orbital speed
    orbSpeed = r0*vphi0/np.sqrt(1+r0*r0*vphi0*vphi0)
    assert (np.isclose(vel.speed(obs), orbSpeed)), "Unexpected orbital speed"

    # The resting observers must all measure the same orbital energy
    orbEnergy = mass*np.sqrt(1+r0*r0*vphi0*vphi0)
    assert (np.isclose(vel.energy(obs), orbEnergy)), "Unexpected orbital energy"
