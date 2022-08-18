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
        if np.sum(G)==0:
            samples['X'] = samples['X'] + sample
            samples['F'] = samples['F'] + F
            samples['G'] = samples['G'] + G

            valid_samples += 1

    return samples


def rand2(problem, mask, n=100):
    """ Performs random sampling with an uniform distribution
    Args:
        problem: a description of the problem for which the surrogate model 
        is going to be built
        n: Number of samples 

    Returns:
        samples: a set of n samples
    """
    x = np.zeros((n, problem.n_var))
    lb = problem.xl # lower bound
    ub = problem.xu # upper bound
    for i in range(0, n): # linhas 
        for j in range(0, problem.n_var): # colunas 
            if mask[j] == "int":
              x[i][j] =   np.random.randint(lb[j], ub[j] + 1)
            else:
              x[i][j] = (lb[j] + np.random.rand()*(ub[j]-lb[j]))



 
    #x = lb + np.random.rand(n,problem.n_var)*(ub-lb)
    if problem.n_constr == 0:
        F = problem.evaluate(x)
        G = []
    else: 
        F,G = problem.evaluate(x)
        #print(F)
    samples = { 'X': x,
                'F': F,
                'G': G}

    return samples
