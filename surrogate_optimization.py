from pymoo.optimize import minimize
import surrogate_selection
import infill_methods
from surrogate_problem import SurrogateProblem

def optimize(problem,optimizer,termination,
                surrogate_ensemble,samples,
                infill_method=infill_methods.rand,
                surrogate_selection_function=surrogate_selection.rand,
                n_infill=1,max_samples=100):

    # Fit surrogates
    obj_surrogates = [fit_surrogate(samples['X'],
                        samples['Y'][i],ensemble=surrogate_ensemble,
                        surrogate_selection_function=surrogate_selection_function) 
                        for i in range(problem.n_obj)]
    const_surrogates = [] if problem.n_constr == 0 else [fit_surrogate(samples['X'],
                        samples['G'][i],ensemble=surrogate_ensemble,
                        surrogate_selection_function=surrogate_selection_function) 
                        for i in range(problem.n_obj)]

    # Build surrogate problem
    surrogate_problem = SurrogateProblem(
        nvar=problem.n_var,
        lb=problem.xl,
        ub=problem.xu,
        obj_surrogate = obj_surrogates,
        const_surrogate = const_surrogates)

    extra_samples = 0
    while extra_samples < max_samples:
        # Optimize for the surrogate problem
        res = minimize(surrogate_problem,
               optimizer,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

        # Compute infill points
        infill_points = infill_method(n_infill,res.X,res.F,res.G)

        # Evaluate infill points


        # Update database
        samples['X'] = samples['X'] + infill_points['X']
        samples['F'] = samples['F'] + infill_points['F']
        samples['G'] = samples['G'] + infill_points['G']

        # Update surrogate problem
        surrogate_problem = update(surrogate_problem,surrogate_ensemble,
                                infill_points,surrogate_selection_function,samples)

        # Update number of extra samples
        extra_samples += n_infill
    
    return res

def fit_surrogate(X,y,ensemble,surrogate_selection_function):
    return surrogate_selection_function(ensemble,X,y).fit(X,y)

def update(surrogate_problem,surrogate_ensemble,
            infill_points,surrogate_selection_function,samples):
    """ Updates the surrogate problem

    """
    selected_obj_surrogate = surrogate_selection_function(
        surrogate_ensemble,
        samples['X'],
        samples['F']) 
    
    selected_const_surrogate = surrogate_selection_function(surrogate_ensemble,
        samples['X'],
        samples['G']) 

    surrogate_problem.obj_surrogate = selected_obj_surrogate
    surrogate_problem.const_surrogate = selected_const_surrogate

    return surrogate_problem

def distance_search_space_infill():
    pass

def distance_objective_space_infill():
    pass