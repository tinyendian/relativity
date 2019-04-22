import numpy as np
import metric as mt

def test_metric():
  metric = mt.metric()
  v1 = np.array([1,2,3,4], dtype = np.float64)
  assert (metric.scalarProduct(v1, v1) == 0), "Expected zero result"

# -----------------------------------------------------------------------

def test_minkowski():
  metric = mt.minkowski()
  matrix = metric.getMatrix()
  assert (np.array_equal(np.diag(matrix), np.array([1,-1,-1,-1], dtype = matrix.dtype))), "Incorrect diagonal"
  assert (np.array_equal(matrix-np.diagflat([1,-1,-1,-1]),
          np.zeros((4,4), dtype = matrix.dtype))), "Non-zero off-diagonal elements"

# -----------------------------------------------------------------------

def test_schwarzschild():
  rs = 1.2345
  r = 2*rs
  theta = 0.5*np.pi
  metric = mt.schwarzschild(rs, r, theta)
  matrix = metric.getMatrix()
  assert (np.isclose(matrix[0,0], 0.5)), "Time component incorrect"
  assert (np.isclose(matrix[1,1], -2)), "Radius component incorrect"
  assert (np.isclose(matrix[2,2], -4*rs*rs)), "Polar angle component incorrect"
  assert (np.isclose(matrix[3,3], -4*rs*rs)), "Azimuth angle component incorrect"
  assert (np.array_equal(matrix-np.diagflat(np.diag(matrix)),
          np.zeros((4,4), dtype = matrix.dtype))), "Non-zero off-diagonal elements"

  rs = 6.54321
  r = 1.0e30
  theta = 1.2345
  metric = mt.schwarzschild(rs, r, theta)
  matrix = metric.getMatrix()
  assert (np.isclose(matrix[0,0], 1)), "Time component does not converge to Minkowski"
  assert (np.isclose(matrix[1,1], -1)), "Radius component does not converge to Minkowski"
