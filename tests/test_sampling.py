from numpy.core.records import array
import pytest
import numpy as np
from pymoo.model.problem import Problem
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import sampling

@pytest.fixture
def test_problem():
  class TestProblem(Problem):
    def __init__(self):
      super().__init__(n_var=2,
                        n_obj=2,
                        n_constr=2,
                        xl=np.array([-2,-2]),
                        xu=np.array([2,2]))

    def _evaluate(self, X, out, *args, **kwargs):
      f1 = X[:,0]**2 + X[:,1]**2
      f2 = (X[:,0]-1)**2 + X[:,1]**2
      
      g1 = 2*(X[:, 0]-0.1) * (X[:, 0]-0.9) / 0.18
      g2 = - 20*(X[:, 0]-0.4) * (X[:, 0]-0.6) / 4.8

      out["F"] = np.column_stack([f1, f2])
      out["G"] = np.column_stack([g1, g2])

  return TestProblem()

'''
 Tests for rand
'''
def test_rand_sampling_should_return_a_dict(test_problem):
  assert (type(sampling.rand(test_problem)) is dict )

@pytest.mark.parametrize("variable_name", ['X', 'F', 'G'])
def test_rand_sampling_should_return_dict_with_X_F_G(test_problem, variable_name):
  assert(variable_name in sampling.rand(test_problem))

def test_rand_should_return_50_samples(test_problem):
  assert(len(sampling.rand(test_problem, 50)['X']) == 50)

def test_rand_samples_should_respect_lower_bound(test_problem):
  lower_bound = test_problem.xl
  samples = sampling.rand(test_problem, 1000)
  X_samples = samples['X']
  if (type(X_samples[0]) is np.ndarray):
    sample_smaller_than_lower_bound = False
    for sample in X_samples:
      sample_smaller_than_lower_bound = any(sample < lower_bound)
      if(sample_smaller_than_lower_bound):
        break
    assert(not sample_smaller_than_lower_bound)
  else:
    min_sample = min(X_samples)
    assert(min_sample > lower_bound)

def test_rand_samples_should_respect_upper_bound(test_problem):
  upper_bound = test_problem.xu
  samples = sampling.rand(test_problem, 1000)
  X_samples = samples['X']
  if (type(X_samples[0]) is np.ndarray):
    sample_bigger_than_upper_bound = False
    for sample in X_samples:
      sample_bigger_than_upper_bound = any(sample > upper_bound)
      if(sample_bigger_than_upper_bound):
        break
    assert(not sample_bigger_than_upper_bound)
  else:
    max_sample = max(X_samples)
    assert(max_sample < upper_bound)

'''
 Tests for ranfom_feasible
'''
@pytest.mark.skip(reason="constraint still apresenting errors, the function is unusable")
def test_random_feasible_sampling_should_return_a_dict(test_problem):
  assert (type(sampling.random_feasible(test_problem)) is dict )

@pytest.mark.skip(reason="constraint still apresenting errors, the function is unusable")
@pytest.mark.parametrize("variable_name", ['X', 'F', 'G'])
def test_random_feasible_sampling_should_return_dict_with_X_F_G(test_problem, variable_name):
  assert(variable_name in sampling.random_feasible(test_problem))

@pytest.mark.skip(reason="constraint still apresenting errors, the function is unusable")
def test_random_feasible_should_return_50_samples(test_problem):
  assert(len(sampling.random_feasible(test_problem, 50)['X']) == 50)

@pytest.mark.skip(reason="constraint still apresenting errors, the function is unusable")
def test_random_feasible_samples_should_respect_lower_bound(test_problem):
  lower_bound = test_problem.xl
  samples = sampling.random_feasible(test_problem, 1000)
  X_samples = samples['X']
  if (type(X_samples[0]) is np.ndarray):
    sample_smaller_than_lower_bound = False
    for sample in X_samples:
      sample_smaller_than_lower_bound = any(sample < lower_bound)
      if(sample_smaller_than_lower_bound):
        break
    assert(not sample_smaller_than_lower_bound)
  else:
    min_sample = min(X_samples)
    assert(min_sample > lower_bound)

@pytest.mark.skip(reason="constraint still apresenting errors, the function is unusable")
def test_random_feasible_samples_should_respect_upper_bound(test_problem):
  upper_bound = test_problem.xu
  samples = sampling.random_feasible(test_problem, 1000)
  X_samples = samples['X']
  if (type(X_samples[0]) is np.ndarray):
    sample_bigger_than_upper_bound = False
    for sample in X_samples:
      sample_bigger_than_upper_bound = any(sample > upper_bound)
      if(sample_bigger_than_upper_bound):
        break
    assert(not sample_bigger_than_upper_bound)
  else:
    max_sample = max(X_samples)
    assert(max_sample < upper_bound)