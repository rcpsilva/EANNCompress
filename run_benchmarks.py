from copy import Error
import sampling
import benchmarks
import numpy as np
import surrogate_optimization
import surrogate_selection
import infill_methods
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.util.plotting import plot
import time
import copy


if __name__ == "__main__":

    # Define original problem
    # problem = benchmarks.mw1()

    # Define original problem
    problem = benchmarks.zdt3()

    # Sample
    randomSample = sampling.rand(problem, 70)

    # Define surrogate ensemble
    

    # Define Optimizer
    optimizer = NSGA2(
        pop_size=250,
        n_offsprings=100,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    # Define termination criteria

    termination = get_termination("n_gen", 100)

    # Define infill criteria

    infill_methods_functions = [
        infill_methods.distance_search_space,
        infill_methods.distance_objective_space,
        infill_methods.rand,
        ]

    # Define surrogate selection

    surrogate_selection_functions = [
        # surrogate_selection.mse,
        # surrogate_selection.mape,
        # surrogate_selection.r2,
        surrogate_selection.spearman,
        surrogate_selection.rand,
    ]
    # Optimize 
    for j, surrogate_selection_function in enumerate(surrogate_selection_functions):
        sampled = []
        for i,infill_method in enumerate(infill_methods_functions):
            start = time.time()
            samples = copy.deepcopy(randomSample)
            surrogate_ensemble = [
                LinearRegression(),
                KNeighborsRegressor(),
            ]
            print('index das parada -----------------------')
            print(i)
            print(j)
            res = surrogate_optimization.optimize(problem,optimizer,termination,
                                surrogate_ensemble,samples,infill_method,
                                surrogate_selection_function,n_infill=2,
                                max_samples=200)
            end = time.time()

            print(samples['X'].shape)
            print(samples['F'].shape)
            print(samples['G'].shape)

            print('Elapsed time: {}'.format(end-start))
            sampled.append(samples)
            

            print(samples['X'].shape)
        
        plt.plot(sampled[0]['F'][:,0],sampled[0]['F'][:,1],'ob')
        plt.plot(sampled[1]['F'][:,0],sampled[1]['F'][:,1],'sg')
        plt.plot(sampled[2]['F'][:,0],sampled[2]['F'][:,1],'xm')
        plot(problem.pareto_front(), no_fill=True)
        plt.show()

        plt.plot(randomSample['F'][:,0],randomSample['F'][:,1],'ob')
        plot(problem.pareto_front(), no_fill=True)
        plt.show()