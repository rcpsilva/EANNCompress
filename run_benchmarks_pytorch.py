from copy import Error
import sampling
#import benchmarks
import numpy as np
import surrogate_optimization
import surrogate_selection
import infill_methods
from compressPyTorch import problem_compress 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.util.plotting import plot
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
import time
import copy
from pymoo.factory import get_problem
import pickle


if __name__ == "__main__":
    
    # Define original problem
    # problem = benchmarks.mw1()

    # Define original problem
    problem_compress = problem_compress.problemResnet() #MW1
    
    #problem = get_problem("zdt3")
    #print(problem.pareto_front())

    # Sample
    mask = ["int", "int", "real", "real", "int", "int", "int"]
    #randomSample = sampling.rand2(problem, mask, 50)
    #randomSample = sampling.rand2(problem, mask, 50)
    #filename2 = 'amostragem'
    #outfile = open(filename2,'wb')
    #pickle.dump(randomSample,outfile)
    #outfile.close()
    filename3 = "amostragem"
    infile = open(filename3,'rb')
    randomSample = pickle.load(infile)
    infile.close()
    
    

    # Define surrogate ensemble
    surrogate_ensemble = [DecisionTreeRegressor(),
        LinearRegression(),
        KNeighborsRegressor()]

    # Define Optimizer
    

    sampling = MixedVariableSampling(mask, {
        "real": get_sampling("real_random"),
        "int": get_sampling("int_random")
    })

    crossover = MixedVariableCrossover(mask, {
        "real": get_crossover("real_sbx", prob=0.9, eta=15),
        "int": get_crossover("int_sbx", prob=0.9, eta=15)
    })

    mutation = MixedVariableMutation(mask, {
        "real": get_mutation("real_pm", eta=20),
        "int": get_mutation("int_pm", eta=20)
    })

    optimizer = NSGA2(
        pop_size=10,
        n_offsprings=10,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    '''
    optimizer = NSGA2(
        pop_size=10,
        n_offsprings=8,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )
    '''
    
    # Define termination criteria

    termination = get_termination("n_gen", 1000)

    # Define infill criteria

    infill_methods_functions = [ # estrategia 
        infill_methods.distance_search_space,
        #infill_methods.distance_objective_space,
        #infill_methods.rand,
        ]

    # Define surrogate selection

    surrogate_selection_functions = [
        surrogate_selection.mse,
        surrogate_selection.mape,
        # surrogate_selection.r2,
        surrogate_selection.spearman,
        surrogate_selection.rand,
    ]

    surrogate_selection_functions2 = [
        surrogate_selection.mse,
        surrogate_selection.mape,
        #surrogate_selection.r2,
        surrogate_selection.spearman,
        surrogate_selection.rand,
    ]
   
    # Optimize 
   # Optimize 
    for j, surrogate_selection_function in enumerate(surrogate_selection_functions):
        sampled = []
        for i,infill_method in enumerate(infill_methods_functions):
            start = time.time()
            samples = copy.deepcopy(randomSample)
            surrogate_ensemble = [
                LinearRegression(),
                KNeighborsRegressor(),
            ]
            print('index das parada -----------------------')
            print(i)
            print(j)
            res = surrogate_optimization.optimize(problem_compress,optimizer,termination,
                                surrogate_ensemble,samples,infill_method,
                                surrogate_selection_function, surrogate_selection.r2,
                                n_infill=2,
                                max_samples=50)
            end = time.time()

            print(samples['X'].shape)
            print(samples['F'].shape)
            print(samples['G'].shape)

            print('Elapsed time: {}'.format(end-start))
            sampled.append(samples)
            

            print(samples['X'].shape)
        
        filename = 'substituto3'
        outfile = open(filename,'wb')
        pickle.dump(sampled,outfile)
        outfile.close()

        plt.plot(sampled[0]['F'][:,0],sampled[0]['F'][:,1],'ob')
        #plt.plot(sampled[1]['F'][:,0],sampled[1]['F'][:,1],'sg')
        #plt.plot(sampled[2]['F'][:,0],sampled[2]['F'][:,1],'xm')
        #plot(problem.pareto_front(), no_fill=True)
        plt.show()
    
        plt.plot(randomSample['F'][:,0],randomSample['F'][:,1],'ob')
        #plot(problem.pareto_front(), no_fill=True)
        plt.show()

        #print(problem.pareto_front())


        '''
         6998.523419141769
         Elapsed time: 12310.416229486465
         '''