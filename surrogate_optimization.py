from pymoo.optimize import minimize

def surrogate_optimize(optimizer,termination,surrogate_ensemble,
                            infill_criteria,n_infill=1,max_samples=100):

    # Select surrogates

    # Build surrogate_problem

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

        # Update surrogate problem
        surrogate_problem = update(surrogate_ensemble,infill_points)

        # Update number of extra samples
        extra_samples += n_infill

def update(surrogate_problem, infill_points):
    pass

def distance_search_space_infill():
    pass

def distance_objective_space_infill():
    pass