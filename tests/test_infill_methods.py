from numpy.core.records import array
import pytest
import numpy as np
from pymoo.model.problem import Problem
import os
import sys
import inspect
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import infill_methods

@pytest.fixture
def test_samples():
  x = np.random.rand(100,3) * 10 + 10
  y = np.array(x[:,0]**2 + x[:,1]**2 + x[:,2]**2 + 10)
  return [x, y]

@pytest.fixture
def know_points():
  A = np.random.rand(15,3) * 10 + 10
  Apf = np.array(A[:,0]**2 + A[:,1]**2 + A[:,2]**2 + 10)
  return [A, Apf]

@pytest.fixture
def five_farthest_points():
  X_far = np.random.rand(5,3) * 200 + 300
  Y_far = np.array(X_far[:,0]**2 + X_far[:,1]**2 + X_far[:,2]**2 + 10)
  return [X_far, Y_far]

@pytest.fixture
def non_dominated_points():
  X_non_dominated = np.array([[1,2,3], [3,2,1]])
  Y_non_dominated = np.array(X_non_dominated[:,0]**2 + X_non_dominated[:,1]**2 + X_non_dominated[:,2]**2 + 10)
  return [X_non_dominated, Y_non_dominated]
'''
 Tests for rand
'''
def test_rand_infill_should_return_a_dict(test_samples, know_points):
  [x, y] =test_samples
  [A, Apf] = know_points
  n = 10
  assert (type(infill_methods.rand(n, x, y, G=[])) is dict )

@pytest.mark.parametrize("variable_name", ['X', 'F', 'G'])
def test_rand_infill_should_return_dict_with_X_F_G(variable_name, test_samples, know_points):
  [x, y] =test_samples
  [A, Apf] = know_points
  n = 10
  assert(variable_name in infill_methods.rand(n, x, y, G=[]))

def test_rand_infill_should_return_30_samples(test_samples, know_points):
  [x, y] =test_samples
  [A, Apf] = know_points
  n = 30
  assert(len(infill_methods.rand(n, x, y, G=[])['X'])==30)

def test_rand_infill_should_return_10_samples_from_test_samples(test_samples, know_points):
  [x, y] = test_samples
  [A, Apf] = know_points
  n = 10
  every_point_is_from_sample = False
  for returned_sample in infill_methods.rand(n, x, y, G=[])['X'] :
    for X in x :
      every_point_is_from_sample = np.array_equal(returned_sample, X)
      if(every_point_is_from_sample): break
    if(not every_point_is_from_sample): break
  assert(every_point_is_from_sample)

'''
 Tests for distance_search_space
'''
def test_distance_search_space_infill_should_return_a_dict(test_samples, know_points):
  [x, y] =test_samples
  [A, Apf] = know_points
  n = 10
  assert (type(infill_methods.distance_search_space(n, x, y, [], A, Apf)) is dict )

@pytest.mark.parametrize("variable_name", ['X', 'F', 'G'])
def test_distance_search_space_infill_should_return_dict_with_X_F_G(variable_name, test_samples, know_points):
  [x, y] =test_samples
  [A, Apf] = know_points
  n = 10
  assert(variable_name in infill_methods.distance_search_space(n, x, y, [], A, Apf))

def test_distance_search_space_infill_should_return_30_samples(test_samples, know_points):
  [x, y] =test_samples
  [A, Apf] = know_points
  n = 30
  assert(len(infill_methods.distance_search_space(n, x, y, [], A, Apf)['X'])==30)

def test_distance_search_space_infill_should_return_10_samples_from_test_samples(test_samples, know_points):
  [x, y] = test_samples
  [A, Apf] = know_points
  n = 10
  every_point_is_from_sample = False
  for returned_sample in infill_methods.distance_search_space(n, x, y, [], A, Apf)['X'] :
    for X in x :
      every_point_is_from_sample = np.array_equal(returned_sample, X)
      if(every_point_is_from_sample): break
    if(not every_point_is_from_sample): break
  assert(every_point_is_from_sample)

def test_distance_search_space_infill_should_return_5_best_samples(test_samples, know_points, five_farthest_points):
  [x, y] = test_samples
  [A, Apf] = know_points
  [X_far, Y_far] = five_farthest_points
  x = np.append(x, X_far, axis=0)
  y= np.append(y, Y_far, axis=0) 
  n = 5
  every_point_is_from_sample = False
  for returned_sample in infill_methods.distance_search_space(n, x, y, [], A, Apf)['X'] :
    for X in X_far :
      every_point_is_from_sample = np.array_equal(returned_sample, X)
      if(every_point_is_from_sample): break
    if(not every_point_is_from_sample): break
  assert(every_point_is_from_sample)


'''
 Tests for distance_objective_space
'''
def test_distance_objective_space_infill_should_return_a_dict(test_samples, know_points):
  [x, y] =test_samples
  [A, Apf] = know_points
  n = 10
  assert (type(infill_methods.distance_objective_space(n, x, y, [], A, Apf)) is dict )

@pytest.mark.parametrize("variable_name", ['X', 'F', 'G'])
def test_distance_objective_space_infill_should_return_dict_with_X_F_G(variable_name, test_samples, know_points):
  [x, y] =test_samples
  [A, Apf] = know_points
  n = 10
  assert(variable_name in infill_methods.distance_objective_space(n, x, y, [], A, Apf))

def test_distance_objective_space_infill_should_return_30_samples(test_samples, know_points):
  [x, y] =test_samples
  [A, Apf] = know_points
  n = 30
  assert(len(infill_methods.distance_objective_space(n, x, y, [], A, Apf)['X'])==30)

def test_distance_objective_space_infill_should_return_2_best_samples(test_samples, know_points, non_dominated_points):
  [x, y] =test_samples
  [A, Apf] = know_points
  [X_non, Y_non] =non_dominated_points
  n = 30
  x = np.append(x, X_non, axis=0)
  y= np.append(y, Y_non, axis=0) 
  n = 2
  every_point_is_from_sample = False
  for returned_sample in infill_methods.distance_objective_space(n, x, y, [], A, Apf)['X'] :
    for X in X_non :
      every_best_point_was_returned = np.array_equal(returned_sample, X)
      if(every_best_point_was_returned): break
    if(not every_best_point_was_returned): break
  assert(every_best_point_was_returned)

# def test_distance_objective_space_infill_should_return_5_best_samples(test_samples, know_points, five_farthest_points):
#   [x, y] = test_samples
#   [A, Apf] = know_points
#   [X_far, Y_far] = five_farthest_points

