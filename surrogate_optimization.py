from pymoo.optimize import minimize
import numpy as np
import surrogate_selection
import infill_methods
from surrogate_problem import SurrogateProblem
import sampling

def optimize(problem,optimizer,termination,
                surrogate_ensemble,samples,
                infill_method=infill_methods.rand,
                surrogate_selection_function=surrogate_selection.rand,
                n_infill=1,max_samples=100):

    surrogate_problem = get_surrogate_problem(problem,
                            samples,
                            surrogate_ensemble,
                            surrogate_selection_function)

    extra_samples = 0
    while extra_samples < max_samples:
        # Optimize for the surrogate problem
        res = minimize(surrogate_problem,
               optimizer,
               termination,
               #seed=1,
               save_history=True,
               verbose=True)

        # Compute and Evaluate infill points
        if res.X is not None:
            actual_n_infill = np.min([res.X.shape[0],n_infill])
            non_dominated = nondominated(samples['X'],samples['F'])

            if problem.n_constr == 0:
                infill_points = infill_method(actual_n_infill,res.X,res.F,[],non_dominated['A'],non_dominated['Apf'])
            else: 
                infill_points = infill_method(actual_n_infill,res.X,res.F,res.G,non_dominated['A'],non_dominated['Apf'])
        else: 
            actual_n_infill = n_infill
            infill_points = {'X': problem.xl + np.random.rand(actual_n_infill,problem.n_var)*(problem.xu-problem.xl)}
        
        if problem.n_constr == 0:
            F = problem.evaluate(infill_points['X'])
            G = []
        else: 
            F,G = problem.evaluate(infill_points['X'])

        # Update database
        samples['X'] = np.vstack((samples['X'], infill_points['X']))
        samples['F'] = np.vstack((samples['F'], F))
        samples['G'] = np.vstack((samples['G'], G))

        # Update surrogate problem
        surrogate_problem = get_surrogate_problem(problem,
                            samples,
                            surrogate_ensemble,
                            surrogate_selection_function)

        # Update number of extra samples
        extra_samples += actual_n_infill
    
    return res

def get_surrogate_problem(problem,samples,surrogate_ensemble,surrogate_selection_function):
        # Fit surrogates
    obj_surrogates = [fit_surrogate(samples['X'],
                        samples['F'][:,i],ensemble=surrogate_ensemble,
                        surrogate_selection_function=surrogate_selection_function) 
                        for i in range(problem.n_obj)]
    const_surrogates = [] if problem.n_constr == 0 else [fit_surrogate(samples['X'],
                        samples['G'][:,i],ensemble=surrogate_ensemble,
                        surrogate_selection_function=surrogate_selection_function) 
                        for i in range(problem.n_constr)]

    # Build surrogate problem
    surrogate_problem = SurrogateProblem(
        nvar=problem.n_var,
        lb=problem.xl,
        ub=problem.xu,
        obj_surrogates = obj_surrogates,
        const_surrogates = const_surrogates)

    return surrogate_problem

def fit_surrogate(X,y,ensemble,surrogate_selection_function):
    model = surrogate_selection_function(ensemble,X,y) 
    return model.fit(X,y)

def nondominated(X,F): # This is very slow, improve in the next iteration
    
    A = None
    Apf = None
    
    for i in range(len(F)):
        non_dom = True
        for j in range(len(F)):
            if i != j:
                if (F[i] >= F[j]).all():
                    non_dom = False
                    break
        if non_dom:
            if A is not None:
                A = np.vstack((A,X[i]))
                Apf = np.vstack((Apf,F[i]))
                
            else:
                A = np.array(X[i])
                Apf = np.array(F[i])

    return {'A': A, 'Apf': Apf}



                 

