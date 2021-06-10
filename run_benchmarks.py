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


if __name__ == "__main__":

    # Define original problem
    problem = benchmarks.mw1()

    # Define original problem
    # problem = benchmarks.zdt1()

    # Sample
    samples = sampling.rand(problem, 100000)

    # Define surrogate ensemble
    surrogate_ensemble = [DecisionTreeRegressor(),
        LinearRegression(),
        KNeighborsRegressor()]

    # Define Optimizer
    optimizer = NSGA2(
        pop_size=50,
        n_offsprings=50,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    # Define termination criteria

    termination = get_termination("n_gen", 1000)

    # Define infill criteria

    infill_method = infill_methods.rand

    # Define surrogate selection

    surrogate_selection_function = surrogate_selection.rand

    # Optimize 
    res = surrogate_optimization.optimize(problem,optimizer,termination,
                        surrogate_ensemble,samples,infill_method,
                        surrogate_selection_function,n_infill=10,
                        max_samples=500)

    print(samples['X'].shape)
    print(samples['F'].shape)
    print(samples['G'].shape)

    plt.plot(samples['F'][:,0],samples['F'][:,1],'o')
    plt.show()

    print(samples['X'].shape)