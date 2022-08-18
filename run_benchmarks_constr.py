from copy import Error
from select import select
import sampling
#import benchmarks
import numpy as np
import surrogate_optimization
import surrogate_selection
import infill_methods
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.util.plotting import plot
from pymoo.optimize import minimize
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.util.ref_dirs import das_dennis
#from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.igd import IGD
from pymoo.factory import get_performance_indicator
from pymoo.indicators.hv import Hypervolume 
#from compressPyTorch import problem_compress 
import pandas as pd

import time
import copy
from pymoo.factory import get_problem
import pickle


if __name__ == "__main__":
    
    # Define original problem
    # problem = benchmarks.mw1()
   
    # Define original problem
    problem = get_problem("mw1") #MW1
    pf = get_problem("mw1").pareto_front()
    
    
    #ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_points=50)
    filename3 = "ref"
    infile = open(filename3,'rb')
    ref_dirs = pickle.load(infile)
    infile.close()
    #randomSamples = sampling.rand(problem, 50)

    termination = get_termination("n_gen", 1000)
    optimizer = CTAEA(ref_dirs=ref_dirs)

    res2 = minimize(problem,
               optimizer,
               ('n_gen', 250),
               seed=1,
               verbose=True
               )
    tamanho = res2.F.shape[0]
    print(tamanho)
    randomSample2 = sampling.rand(problem, (50 - tamanho))
    randomSample = { 'X': np.concatenate((res2.X, randomSample2['X']), axis=0),
                'F': np.concatenate((res2.F, randomSample2['F']), axis=0),
                'G': np.concatenate((res2.G, randomSample2['G']), axis=0)}
    #randomSample['F'].append(randomSample2['F'])

    #randomSample = np.concatenate((samples2, randomSample2))

    surrogate_ensemble = [
                LinearRegression(),
                KNeighborsRegressor(),
            ]
    infill_methods_functions = [ # estrategia 
        infill_methods.distance_search_space,
        #infill_methods.distance_objective_space,
        #infill_methods.rand,
        ]

    # Define surrogate selection

    surrogate_selection_functions = [
        surrogate_selection.mse,
        surrogate_selection.mape,
        #surrogate_selection.r2,
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
    ref_point = [np.max(randomSample['F'][0]), np.max(randomSample['F'][1]) ]#, np.max(randomSample['F'][1] ))]
    ind2 = Hypervolume(ref_point=ref_point)
    k = np.where(randomSample['G'] <= 0.0)[0]
    print(ind2.do(randomSample['F'][k]))
    print("shape fora", k.shape)
    '''
    k = np.where(randomSample['G'] <= 0.0)[0]
    ll = randomSample['F'][k]
    print(np.max(ll[0]))
    print(k)
    '''
    select_restricao = [] 
    select_objetivo = []
    estrategia = []
    list_idg = []
    list_idgT = []
    list_hp = []
    list_hpT = []
    num_infil = []
    parada = []
    algoritmo = []
    factivel = []
    cont = 0
    
    for j, surrogate_selection_function in enumerate(surrogate_selection_functions): # objetivo
        for k, surrogate_selection_function2 in enumerate(surrogate_selection_functions2): # retricao
            for i,infill_method in enumerate(infill_methods_functions): # distance_search_space
                for z in range(0,1): # repetições
                    samples = copy.deepcopy(randomSample)
                    surrogate_ensemble = [
                            LinearRegression(),
                            KNeighborsRegressor(),
                    ]
                    res = surrogate_optimization.optimize(problem,optimizer,termination,
                                surrogate_ensemble,samples,infill_method,
                                surrogate_selection_function, surrogate_selection_function2,
                                n_infill=2,
                                max_samples=50)
                    print("uma execucao")
                    k = np.where(samples['G'] <= 0.0)[0] # soluçõe sque atendem as restrições
                    print("shape das factiveis", k.shape)
                    ind = get_performance_indicator("igd", pf)
                    ref_point = [np.max(samples['F'][0]), np.max(samples['F'][1]) ]
                    ind2 = Hypervolume(ref_point=ref_point)
                    select_restricao.append(surrogate_selection_function2.__name__)
                    select_objetivo.append(surrogate_selection_function.__name__) 
                    estrategia.append(infill_method.__name__)
                    list_idg.append(ind.do(samples['F'][k]))
                    list_idgT.append(ind.do(samples['F']))
                    list_hp.append(ind2.do(samples['F'][k]))
                    list_hpT.append(ind2.do(samples['F']))
                    num_infil.append(2)
                    parada.append(1000)
                    factivel.append(k.shape[0])
                    algoritmo.append("CTAEA")
                    cont = cont + 1
                    print(cont)

    print("terminou")
    dados = {
        'select_restricao' :select_restricao, 
        'select_objetivo' : select_objetivo,
        'estrategia' : estrategia,
        'list_idg' : list_idg,
        'list_hp' : list_hp,
        'num_infil' : num_infil,
        'parada' : parada,
        'algoritmo' : algoritmo,
        'hvT' : list_idgT,
        'idgT' : list_hpT,
        'factivel' : factivel


    }

    planilha1 = pd.DataFrame(dados)
    planilha1.to_csv('planilha2.csv')
    
    '''
    res = surrogate_optimization.optimize(problem,optimizer,termination,
                                surrogate_ensemble,samples,infill_methods.distance_search_space,
                                surrogate_selection.mse, surrogate_selection.r2,
                                n_infill=2,
                                max_samples=50)

   
    #print(samples['X'])
    #print(samples['F'])
    #print(samples['G'])
                           
    
    res = minimize(problem,
               optimizer,
               ('n_gen', 250),
               seed=1,
               verbose=True
               )

    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)
    print("Constraint violation: %s" % res.CV)
    
    #ind = IGD(pf)
    #print("IGD", ind(res.F))
    pf = get_problem("zdt1").pareto_front()

# The result found by an algorithm
    A = pf[::10] * 1.1
    ind = get_performance_indicator("igd", pf)
    
    print("IGD", ind.do(A))
    
    ref_point = np.array([1.2, 1.2])

    ind2 = Hypervolume(ref_point=ref_point)
    print("HV", ind2.do(A))

    select_restricao = [] 
    select_objetivo = []
    estrategia = []
    list_idg = []
    list_hp = []
    num_infil = []
    parada = []
    algoritmo = []
    

    for i in range(4):
        select_restricao.append(surrogate_selection_functions2[i].__name__)
        select_objetivo.append(surrogate_selection_functions[i].__name__) 
        estrategia.append(infill_methods_functions[0].__name__)
        list_idg.append(ind.do(A))
        list_hp.append(ind2.do(A))
        num_infil.append(2)
        parada.append(250)
        algoritmo.append("NSGA2")


    dados = {
        'select_restricao' :select_restricao, 
        'select_objetivo' : select_objetivo,
        'estrategia' : estrategia,
        'list_idg' : list_idg,
        'list_hp' : list_hp,
        'num_infil' : num_infil,
        'parada' : parada,
        'algoritmo' : algoritmo

    }

    pessoas = pd.DataFrame(dados)
    pessoas.to_csv('pessoas.csv')
     '''



    