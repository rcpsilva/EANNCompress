from pymoo.optimize import minimize
import surrogate_selection

def surrogate_optimize(optimizer,termination,surrogate_problem,
                        surrogate_ensemble,samples,infill_criteria,
                        surrogate_selection_function=surrogate_selection.select_random,n_infill=1,max_samples=100):

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
        infill_points = infill_criteria(res.X,res.F,n_infill)

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