import sampling
import benchmarks
import numpy as np
import surrogate_optimization
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

if __name__ == "__main__":

    # Define original problem
    problem = benchmarks.mw1()

    # Sample
    samples = sampling.random(problem, 15)

    # Define surrogate ensemble
    surrogate_ensemble = [DecisionTreeRegressor(),
        LinearRegression(),
        KNeighborsRegressor()]

    # Optimize 

    print(samples['X'])
    print(samples['F'])
    print(samples['G'])
