import numpy as np
import transformation as tf

def test_transformation():
  transformation = tf.transformation()
  v = np.ones(4, dtype = np.float64)
  assert (np.array_equal(transformation.transformContraVector(v),
          np.zeros_like(v))), "Expected zero result"
  m = np.ones((4,4), dtype = np.float64)
  assert (np.array_equal(transformation.transformCoMatrix(m),
          np.zeros_like(m))), "Expected zero result"

# -----------------------------------------------------------------------

def test_lorentzRotation():

  # Define contravariant vectors for each dimension
  e0 = np.array([1,0,0,0], dtype = np.float64)
  e1 = np.array([0,1,0,0], dtype = np.float64)
  e2 = np.array([0,0,1,0], dtype = np.float64)
  e3 = np.array([0,0,0,1], dtype = np.float64)

  # Rotate vectors around each axis
  angle = 0.5*np.pi
  for axis in range(1,4):
    rot = tf.lorentzRotation(axis, angle)
    assert (np.isclose(np.linalg.det(rot.getMatrix()),1)), "Expected determinant 1"
    assert (np.isclose(np.linalg.det(rot.getInvMatrix()),1)), "Expected determinant 1"
    assert (np.allclose(np.matmul(rot.getMatrix(),rot.getInvMatrix()),np.eye(4, dtype = np.float64))), "Expected diagonal matrix"
    
    if axis == 1:
      assert (np.allclose(rot.transformContraVector(e0), e0)), "Time component is not invariant"
      assert (np.allclose(rot.transformContraVector(e1), e1)), "Component on rot axis is not invariant"
      assert (np.allclose(rot.transformContraVector(e2), e3)), "Unexpected result"
      assert (np.allclose(rot.transformContraVector(e3), -e2)), "Unexpected result"
    elif axis == 2:
      assert (np.allclose(rot.transformContraVector(e0), e0)), "Time component is not invariant"
      assert (np.allclose(rot.transformContraVector(e1), -e3)), "Unexpected result"
      assert (np.allclose(rot.transformContraVector(e2), e2)), "Component on rot axis is not invariant"
      assert (np.allclose(rot.transformContraVector(e3), e1)), "Unexpected result"
    elif axis == 3:
      assert (np.allclose(rot.transformContraVector(e0), e0)), "Time component is not invariant"
      assert (np.allclose(rot.transformContraVector(e1), e2)), "Unexpected result"
      assert (np.allclose(rot.transformContraVector(e2), -e1)), "Unexpected result"
      assert (np.allclose(rot.transformContraVector(e3), e3)), "Component on rot axis is not invariant"

  # Zero rotations
  angle = 0
  for axis in range(1,4):
    rot = tf.lorentzRotation(axis, angle)
    assert (np.allclose(rot.getMatrix(),np.eye(4, dtype = np.float64)) == True), "Expected diagonal matrix"
    assert (np.allclose(rot.getInvMatrix(),np.eye(4, dtype = np.float64)) == True), "Expected diagonal matrix"
  
  # Inverse rotation
  angle = 0.123
  for axis in range(1,4):
    rot = tf.lorentzRotation(axis, angle)
    rotInv = tf.lorentzRotation(axis, -angle)
    assert (np.allclose(np.matmul(rot.getMatrix(),rotInv.getMatrix()),np.eye(4, dtype = np.float64))), "Expected diagonal matrix"

  # Minkowski metric must be invariant under rotation
  m = np.diagflat([1,-1,-1,-1])
  angle = 0.5*np.pi
  for axis in range(1,4):
    rot = tf.lorentzRotation(axis, angle)
    assert (np.allclose(rot.transformCoMatrix(m), m)), "Minkowski metric not invariant under rotation"

# -----------------------------------------------------------------------

def test_lorentzBoost():

  # Define contravariant vectors for each dimension
  e0 = np.array([1,0,0,0], dtype = np.float64)
  e1 = np.array([0,1,0,0], dtype = np.float64)
  e2 = np.array([0,0,1,0], dtype = np.float64)
  e3 = np.array([0,0,0,1], dtype = np.float64)

  # Boost with half lightspeed
  beta = 0.5
  gamma = 1.0/np.sqrt(1-beta*beta)

  # Boost along each axis
  for axis in range(1,4):
    boost = tf.lorentzBoost(axis, beta)
    assert (np.isclose(np.linalg.det(boost.getMatrix()),1)), "Expected determinant 1"
    assert (np.isclose(np.linalg.det(boost.getInvMatrix()),1)), "Expected determinant 1"
    assert (np.allclose(np.matmul(boost.getMatrix(), boost.getInvMatrix()),np.eye(4, dtype = np.float64))), "Expected diagonal matrix"
    
    if axis == 1:
      assert (np.allclose(boost.transformContraVector(e0), gamma*e0-beta*gamma*e1)), "Time component is not boosted"
      assert (np.allclose(boost.transformContraVector(e1), -beta*gamma*e0+gamma*e1)), "x1 component is not boosted"
      assert (np.allclose(boost.transformContraVector(e2), e2)), "x2 component is boosted"
      assert (np.allclose(boost.transformContraVector(e3), e3)), "x3 component is boosted"
    elif axis == 2:
      assert (np.allclose(boost.transformContraVector(e0), gamma*e0-beta*gamma*e2)), "Time component is not boosted"
      assert (np.allclose(boost.transformContraVector(e1), e1)), "x1 component is boosted"
      assert (np.allclose(boost.transformContraVector(e2), -beta*gamma*e0+gamma*e2)), "x2 component is not boosted"
      assert (np.allclose(boost.transformContraVector(e3), e3)), "x3 component is boosted"
    elif axis == 3:
      assert (np.allclose(boost.transformContraVector(e0), gamma*e0-beta*gamma*e3)), "Time component is not boosted"
      assert (np.allclose(boost.transformContraVector(e1), e1)), "x1 component is boosted"
      assert (np.allclose(boost.transformContraVector(e2), e2)), "x2 component is boosted"
      assert (np.allclose(boost.transformContraVector(e3), -beta*gamma*e0+gamma*e3)), "x3 component is not boosted"

  # Zero boosts
  beta = 0
  for axis in range(1,4):
    boost = tf.lorentzBoost(axis, beta)
    assert (np.allclose(boost.getMatrix(),np.eye(4, dtype = np.float64))), "Expected diagonal matrix"
    assert (np.allclose(boost.getInvMatrix(),np.eye(4, dtype = np.float64))), "Expected diagonal matrix"

  # Inverse boost
  beta = 0.8
  for axis in range(1,4):
    boost = tf.lorentzBoost(axis, beta)
    boostInv = tf.lorentzBoost(axis, -beta)
    assert (np.allclose(np.matmul(boost.getMatrix(),boostInv.getMatrix()),np.eye(4, dtype = np.float64))), "Expected diagonal matrix"

  # Minkowski metric must be invariant
  m = np.diagflat([1,-1,-1,-1])
  beta = -0.654
  for axis in range(1,4):
    boost = tf.lorentzBoost(axis, beta)
    assert (np.allclose(boost.transformCoMatrix(m), m)), "Minkowski metric not invariant"
