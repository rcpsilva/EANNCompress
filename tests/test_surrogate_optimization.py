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
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.util.plotting import plot

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import surrogate_optimization

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
def optimizer():
  return NSGA2(
        pop_size=5,
        n_offsprings=7,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )
@pytest.fixture
def termination():
  return get_termination("n_gen", 7)

@pytest.fixture
def mock_surrogate_selection_function():
  def surrogate_fake_selection(*args):
    return LinearRegression() 
  return surrogate_fake_selection

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

def test_nondominated_should_return_only_nondominated_samples():
  X = np.array([1,      2,      3,      4,     5,     6,    7,      8,      9,      10])
  F =np.array([[2,6],[10,10],[11, 11],[1,10],[9,1],[2,3],[10,10],[10,10],[10,10],[10,10]])
  X_nondominated = np.array([4,5,6])
  F_nondominated = np.array([[1,10], [9,1], [2,3]])
  non_dominated = surrogate_optimization.nondominated(X, F)
  print([a in X_nondominated for a in non_dominated['A']])
  assert (
    all([a in X_nondominated for a in non_dominated['A']])
    and all([a in F_nondominated for a in non_dominated['Apf']]))


def test_fit_surrogate_returns_a_trained_model(test_samples, surrogate_ensemble ,mock_surrogate_selection_function):
  [x, y] = test_samples
  assert (type(surrogate_optimization.fit_surrogate(x,y, surrogate_ensemble, mock_surrogate_selection_function)))

