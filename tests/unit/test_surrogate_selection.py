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
grandparentdir =os.path.dirname(parentdir)
sys.path.insert(1, grandparentdir) 

import surrogate_selection

@pytest.fixture
def surrogate_ensemble():
  return [DecisionTreeRegressor(),
        LinearRegression(),
        KNeighborsRegressor()]

@pytest.fixture
def test_samples():
  x = np.random.rand(100,3) * 10
  y = x[:,0]**2 + x[:,1]**2 + x[:,2]**2
  return [x, y]

@pytest.fixture
def linear_function_samples():
  x = np.random.rand(100,1) * 10
  y = x[:,0]*5 + 3
  return [x, y]

'''
Test if selection fucntions are returning one of the ensemble models
'''
def test_rand_should_return_one_model_from_given_ensemble(surrogate_ensemble, test_samples):
  [x, y] = test_samples
  assert surrogate_selection.rand(surrogate_ensemble, x, y) in surrogate_ensemble

def test_mse_should_return_one_model_from_given_ensemble(surrogate_ensemble, test_samples):
  [x, y] = test_samples
  assert surrogate_selection.mse(surrogate_ensemble, x, y) in surrogate_ensemble

def test_mape_should_return_one_model_from_given_ensemble(surrogate_ensemble, test_samples):
  [x, y] = test_samples
  assert surrogate_selection.mape(surrogate_ensemble, x, y) in surrogate_ensemble

def test_r2_should_return_one_model_from_given_ensemble(surrogate_ensemble, test_samples):
  [x, y] = test_samples
  assert surrogate_selection.r2(surrogate_ensemble, x, y) in surrogate_ensemble

def test_spearman_should_return_one_model_from_given_ensemble(surrogate_ensemble, test_samples):
  [x, y] = test_samples
  assert surrogate_selection.spearman(surrogate_ensemble, x, y) in surrogate_ensemble

def test_selection_by_metric_should_return_one_model_from_given_ensemble(surrogate_ensemble, test_samples):
  metric = mean_squared_error
  [x, y] = test_samples
  selected = surrogate_selection.by_metric(surrogate_ensemble, metric, x, y, metric_great_is_better=False)

'''
Test if the fucntions are returning Linear Regression for an linear function
'''

def test_mse_should_return_linear_regression_from_given_ensemble(surrogate_ensemble, linear_function_samples):
  [x, y] = linear_function_samples
  assert type(surrogate_selection.spearman(surrogate_ensemble, x, y)) is type(LinearRegression())

def test_mape_should_return_linear_regression_from_given_ensemble(surrogate_ensemble, linear_function_samples):
  [x, y] = linear_function_samples
  assert type(surrogate_selection.spearman(surrogate_ensemble, x, y)) is type(LinearRegression())

def test_r2_should_return_linear_regression_from_given_ensemble(surrogate_ensemble, linear_function_samples):
  [x, y] = linear_function_samples
  assert type(surrogate_selection.spearman(surrogate_ensemble, x, y)) is type(LinearRegression())

def test_spearman_should_return_linear_regression_from_given_ensemble(surrogate_ensemble, linear_function_samples):
  [x, y] = linear_function_samples
  assert type(surrogate_selection.spearman(surrogate_ensemble, x, y)) is type(LinearRegression())
