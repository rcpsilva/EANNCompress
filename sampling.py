######################################
# Wrapper for sampling methods
######################################

import numpy as np
from pymoo.factory import get_sampling
from pymoo.util import plotting
from pymoo.interface import sample

def rand(problem, n=100):
    """ Performs random sampling with an uniform distribution
    Args:
        problem: a description of the problem for which the surrogate model 
        is going to be built
        n: Number of samples 

    Returns:
        samples: a set of n samples
    """
    lb = problem.xl # lower bound
    ub = problem.xu # upper bound 
    x = lb + np.random.rand(n,problem.n_var)*(ub-lb)
    if problem.n_constr == 0:
        F = problem.evaluate(x)
        G = []
    else: 
        F,G = problem.evaluate(x)
    samples = { 'X': x,
                'F': F,
                'G': G}

    return samples

def random_feasible(problem, n=100):
    """ Performs random sampling with an uniform distribution
    Args:
        problem: a description of the problem for which the surrogate model 
        is going to be built
        n: Number of samples 

    Returns:
        samples: a set of n feasible samples 
    """
    lb = problem.xl # lower bound
    ub = problem.xu # upper bound 
    samples = { 'X': [],
                'F': [],
                'G': []}

    valid_samples = 0

    while valid_samples <= n:
        sample = lb + np.random.rand(1,problem.n_var)*(ub-lb)
        F,G = problem.evaluate(sample[0])
        if np.sum(G)<=0 :  
            samples['X'].append(sample)
            samples['F'].append(F)
            samples['G'].append(G)

            valid_samples += 1

    return samples

def lhs(n):
    sampling = get_sampling('real_lhs' )
    x = sample(sampling, n, 2)  
    F = []
    G = []
    samples = { 'X': x,
                'F': F,
                'G': G}
    return samples
