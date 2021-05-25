import numpy as np

def rand(n,X,F,G=[]):
    random_indices = np.random.choice(X.shape[0], size=n, replace=False)
    nX = X[random_indices,:]
    nF = F[random_indices,:]
    nG = G[random_indices,:] if G else [] 
    return {'X':nX,
            'F':nF,
            'G':nG}

def distance_search_space_indices(n, X, A, X_non_solution_indices = None):
    solutions_indices = []
    min_distances = []
    if(X_non_solution_indices is None):
        X_non_solution_indices = np.arange(len(X))
    
    for i,Xi in enumerate(X):
        distance_vectors = np.subtract(Xi, A)
        Euclidean_distance = np.linalg.norm(distance_vectors, ord=2, axis=1)
        min_distance_index = np.argmin(Euclidean_distance)
        min_distances.append(Euclidean_distance[min_distance_index])
    
    X_solution_index = np.argmax(min_distances)
    solutions_indices.append(X_non_solution_indices[X_solution_index])
    if (n > 1):
        new_A = np.append(A, [X[X_solution_index]], axis=0)
        new_X = np.delete(X, X_solution_index, axis=0)
        X_non_solution_indices = np.delete(X_non_solution_indices, X_solution_index)
        new_solution = distance_search_space_indices(n-1, new_X, new_A, X_non_solution_indices= X_non_solution_indices)
        solutions_indices = np.append(solutions_indices, new_solution)
    
    return solutions_indices

def distance_search_space(n, X, F, A, G=[]):
    solutions_indices = distance_search_space_indices(n, X, A)
    nX = X[solutions_indices,:]
    nF = F[solutions_indices,:]
    nG = G[solutions_indices,:] if G else [] 
    return {'X':nX,
            'F':nF,
            'G':nG}


        
        

def distance_objective_space(n, X, F, Apf, G=[], great_is_better=False):
    no_dominated_non_solution_indices = np.arange(len(F))
    dominated_non_solution_indices = []
    
    F_non_dominated = F
    
    for indice, Fi in enumerate(F):
        for Apfj in Apf:
            if (max(np.subtract(Fi,Apfj))>0):
                dominated_non_solution_indices.append(no_dominated_non_solution_indices[indice])
                no_dominated_non_solution_indices = np.delete(no_dominated_non_solution_indices, indice)
                F_non_dominated = np.delete(F_non_dominated, indice, axis=0)
                break

    n_F_non_dominated =  len(F_non_dominated)
    number_infill_non_random = min(n, n_F_non_dominated)
    non_dominated_solutions = distance_search_space_indices(number_infill_non_random, F_non_dominated, Apf, no_dominated_non_solution_indices)
    
    random_indices =[]
    if(n > n_F_non_dominated):
        number_infill_random = n - n_F_non_dominated
        random_indices = np.random.choice(dominated_non_solution_indices, size=number_infill_random, replace=False)

    solutions_indices = non_dominated_solutions
    solutions_indices = np.append(solutions_indices, random_indices)

    nX = X[solutions_indices,:]
    nF = F[solutions_indices,:]
    nG = G[solutions_indices,:] if G else [] 
    return {'X':nX,
            'F':nF,
            'G':nG}

if __name__ == "__main__":

    X = np.array([[1,1,1],[2,2,2],[3,3,3],])

    A = np.array([[1,2,3],[5,4,3]])

    F = np.array([[10,2,3],[1,2,0],[-20,-4,-30],])
    distance_objective_space(3, X, F, A)
