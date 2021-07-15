import numpy as np

def rand(n,X,F,G=[],A=[],Apf=[]):
    """ Implements a random infill method

    From a set of possible infill points it selects n random points

    Args: 
        n: Number of infill points.
        X: Set of samples.
        F: The objective value for each point.
        G: The constraint value for each point.

    Returns:
        A dict of X, F and G of the selected points.

    """

    
    random_indices = np.random.choice(X.shape[0], size=n, replace=False)
    nX = X[random_indices]
    nF = F[random_indices]
    nG = G[random_indices] if len(G) else [] 
    return {'X':nX,
            'F':nF,
            'G':nG}

def farthest_points_indices(n, X, A, X_non_solution_indices = None):
    """ Returns the n farthest X points indices with relation with A

    From a X set, the function returns the indices of the n points that
    have the maximum euclidean distance with the nearest point of the set A

    Args:
        n: the number of samples indices to return. Must be a positive integer
        X: A set samples of which the points will be chosen
        A: A set of known samples
        X_non_solution_indices: indices of X that cannot be used as a solution.
    
    Returns:
        A set of n indices in order, from the farthest point of X to the 
        nearest. example
        [4,2,16,15]
    """
    solutions_indices = []
    min_distances = []

    if(X_non_solution_indices is None):
        X_non_solution_indices = np.arange(len(X))
    
    axis = 1
    # if the elements of X are numbers some functions from numpy break, like norm, this if will deal with it
    if(type(X[0]) == np.int64 or type(X[0]) == np.float64):
        axis = 0
    for i,Xi in enumerate(X):
        distance_vectors = np.subtract(Xi, A)
        Euclidean_distance = np.linalg.norm(distance_vectors, ord=2, axis=axis)
        if(axis==0):
            min_distances.append(Euclidean_distance)
        else:
            min_distance_index = np.argmin(Euclidean_distance)
            min_distances.append(Euclidean_distance[min_distance_index])
    
    X_solution_index = np.argmax(min_distances)
    solutions_indices.append(X_non_solution_indices[X_solution_index])
    if (n > 1 and len(X)>0):
        new_A = np.append(A, [X[X_solution_index]], axis=0)
        new_X = np.delete(X, X_solution_index, axis=0)
        X_non_solution_indices = np.delete(X_non_solution_indices, X_solution_index)
        new_solution = farthest_points_indices(n-1, new_X, new_A, X_non_solution_indices= X_non_solution_indices)
        solutions_indices = np.append(solutions_indices, new_solution)
    
    return solutions_indices

def distance_search_space(n,X,F,G=[],A=[],Apf=[]):
    """ Sample infill points based on the distance in the search space

    From a set of possible infill points it selects the 'n' with the most distant neighbors

    Args: 
        n: Number of infill points.
        X: Set of samples.
        F: The objective value for each point.
        G: The constraint value for each point.
        A: set of known points
        Apf: The objective value for each point of 'A'

    Returns:
        A dict of X, F and G of the selected points.

    """

    solutions_indices = farthest_points_indices(n, X, A)
    nX = X[solutions_indices]
    nF = F[solutions_indices]
    nG = G[solutions_indices] if len(G) else [] 
    return {'X':nX,
            'F':nF,
            'G':nG}

def distance_objective_space(n, X, F, G=[], A=[], Apf=[]):
    """ Sample infill points based on the distance in the objective space

    From a set of possible infill points it selects 'n' with the most distant and
    non-dominanted neighbors in the objective space, if less than 'n' points are non-dominated
    than the remaining points are randomly selected.

    Args: 
        n: Number of infill points.
        X: Set of samples.
        F: The objective value for each point.
        G: The constraint value for each point.
        A: set of known points
        Apf: The objective value for each point of 'A'

    Returns:
        A dict of X, F and G of the selected points.

    """
    no_dominated_non_solution_indices = np.arange(len(F))
    dominated_non_solution_indices = []
    
    F_non_dominated = F
    
    for indice, Fi in enumerate(F):
        for indiceApf, Apfj in enumerate(Apf):
            distance_vector = np.subtract(Fi,Apfj)
            max_distance_fi_apfj = distance_vector
            if(type(distance_vector) == np.int64 or type(distance_vector) == np.float64):
                max_distance_fi_apfj = distance_vector
            else: 
                max_distance_fi_apfj = max(distance_vector)
            if (max_distance_fi_apfj>0):
                dominated_non_solution_indices.append(no_dominated_non_solution_indices[indice])
                break
    no_dominated_non_solution_indices = np.delete(no_dominated_non_solution_indices, dominated_non_solution_indices)
    F_non_dominated = np.delete(F_non_dominated, dominated_non_solution_indices, axis=0)
    n_F_non_dominated =  len(F_non_dominated)
    number_infill_non_random = min(n, n_F_non_dominated)
    non_dominated_solutions = []
    if number_infill_non_random > 0 :
        non_dominated_solutions = farthest_points_indices(number_infill_non_random, F_non_dominated, Apf, no_dominated_non_solution_indices)
    
    random_indices =[]
    if(n > number_infill_non_random):
        number_infill_random = n - number_infill_non_random
        random_indices = np.random.choice(dominated_non_solution_indices, size=number_infill_random, replace=False)

    solutions_indices = non_dominated_solutions
    solutions_indices = np.append(solutions_indices, random_indices)
    solutions_indices = solutions_indices.astype(int)
    nX = X[solutions_indices]
    nF = F[solutions_indices]
    nG = G[solutions_indices] if len(G) else [] 

    return {'X':nX,
            'F':nF,
            'G':nG}

if __name__ == "__main__":

    X = np.array([[1,1,1],[2,2,2],[3,3,3],])

    A = np.array([[1,2,3],[5,4,3]])

    F = np.array([10,1,0])
    print(distance_search_space(2, X, F, np.array([[1,2,3],[5,4,3],[1,2,5]]), A))
