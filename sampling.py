######################################
# Wrapper for sampling methods
######################################

import numpy as np

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
    print(x)
    print('after----------------------')
    if(hasattr(problem, 'get_compression_mask')):
        x = turn_variables_type_by_mask(x, problem.get_compression_mask())
    print(x)
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
        if(hasattr(problem, 'get_compression_mask')):
            sample = turn_variables_type_by_mask(sample, problem.get_compression_mask())
        if problem.n_constr == 0:
            F = problem.evaluate(sample[0])
            G = []
        else: 
            F,G = problem.evaluate(sample[0])
        if np.sum(G)==0:
            samples['X'] = samples['X'] + sample
            samples['F'].append(F)
            if problem.n_constr > 0:
                samples['G'] = samples['G'] + G
            valid_samples += 1
    return samples


def turn_variables_type_by_mask(X, mask):
    X_masked = []
    for xi in X:
        xi_masked = []
        for j, xij in enumerate(xi):
            if(mask[j] == "int"):
                xi_masked.append(int(round(xij)))
            else:
                xi_masked.append(xij)
        X_masked.append(xi_masked)
    return X_masked
