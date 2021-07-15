from numpy.core.records import array
import pytest
import numpy as np
from pymoo.model.problem import Problem
import os
import sys
import inspect
import copy
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir =os.path.dirname(parentdir)
sys.path.insert(1, grandparentdir) 

import surrogate_optimization
import surrogate_selection
import infill_methods
import sampling
import benchmarks


infill_methods_functions = [
    infill_methods.distance_objective_space,
    infill_methods.distance_search_space,
    infill_methods.rand,
    ]

surrogate_selection_functions = [
    surrogate_selection.mse,
    surrogate_selection.mape,
    surrogate_selection.r2,
    surrogate_selection.spearman,
    surrogate_selection.rand,
    ]



def test_surrogate_optimize():
    # Define original problem
    problem = benchmarks.zdt3()

    # Sample
    randomSample = sampling.rand(problem, 15)

    # Define surrogate ensemble
    

    # Define Optimizer
    optimizer = NSGA2(
        pop_size=20,
        n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    # Define termination criteria

    termination = get_termination("n_gen", 10)

    # Define infill criteria

    infill_method= infill_methods.rand

    # Define surrogate selection

    surrogate_selection_function = surrogate_selection.rand
    # Optimize 
    
    samples = copy.deepcopy(randomSample)
    surrogate_ensemble = [
        LinearRegression(),
        KNeighborsRegressor(),
        DecisionTreeRegressor(),
    ]
    try:
      res = surrogate_optimization.optimize(problem,optimizer,termination,
                        surrogate_ensemble,samples,infill_method,
                        surrogate_selection_function,n_infill=2,
                        max_samples=20)
    except Exception as e:
      raise pytest.fail("DID RAISE {0}".format(e))


@pytest.mark.parametrize("infill", infill_methods_functions)
@pytest.mark.parametrize("surrogate_select_method", surrogate_selection_functions)
def test_surrogate_optimize_with_all_selection_and_infill_methods(infill, surrogate_select_method):
    # Define original problem
    try:
        problem = benchmarks.zdt3()

        # Sample
        randomSample = sampling.rand(problem, 15)

        # Define surrogate ensemble
        

        # Define Optimizer
        optimizer = NSGA2(
            pop_size=20,
            n_offsprings=10,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )

        # Define termination criteria

        termination = get_termination("n_gen", 10)

        # Define infill criteria

        infill_method= infill

        # Define surrogate selection

        surrogate_selection_function = surrogate_select_method
        # Optimize 
        
        samples = copy.deepcopy(randomSample)
        surrogate_ensemble = [
            LinearRegression(),
            KNeighborsRegressor(),
            DecisionTreeRegressor(),
        ]
        res = surrogate_optimization.optimize(problem,optimizer,termination,
                        surrogate_ensemble,samples,infill_method,
                        surrogate_selection_function,n_infill=2,
                        max_samples=20)
    except Exception as e:
      raise pytest.fail("infill: {0} /n and surrogate_selection: {1}/n raise a error: {2}".format(infill, surrogate_select_method, e))

