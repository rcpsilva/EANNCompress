from tabnanny import verbose
import numpy as np
#from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from . import compression
import pickle


class MyProblem(ElementwiseProblem):
    
    def __init__(self, limite):
        super().__init__(n_var=7,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([0, 0, 0, 0, 0, limite[0], 0 ]), #limite inferiror de cada variavel 
                         xu=np.array([2, 2, 1, 1, 2, limite[1], 1])) # limite superior de cada variavel

    def _evaluate(self, x, out, *args, **kwargs):
        # x representa uma unica solução a ser avaliada 
        # out é um dicionario que representa a saída
        
        XX = compression.comprime(x, op = 'resnet50') 
        f1 = -1*(XX[0]) 
        f2 = XX[1] 

        g1 = 74.0 + f1  # restrição
        

        out["F"] = [f1, f2] # saida para os objetivos
        out["G"] = [g1] # saida para as restrições 


#problem = MyProblem()

def problemResnet():
    return MyProblem(limite=[1,1])

'''
problem = problemResnet()

mask = ["int", "int", "real", "real", "int", "int", "int"]

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




algorithm = NSGA2(
    pop_size=10,
    n_offsprings=10,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True,
)
termination = get_termination("n_gen", 10)
res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    save_history=True,
    verbose= False
)

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)

filename = 'original'
outfile = open(filename,'wb')
pickle.dump(res,outfile)
outfile.close()

print("fechou")
'''

'''
5 3 e 5g
 [[0 0 0.03905478323288236 0.1698304195645689 1 1 1]
 [0 2 0.8781425034294131 0.16207817302326008 1 2 1]
 [1 2 0.8772926139954162 0.3450697825068405 0 2 1]]
Function value: [[-78.43         1.        ]
 [-78.42         0.85094329]
 [-78.16         0.68261894]]
Constraint violation: [[0.]
 [0.]
 [0.]]




  [[0 2 0.03225221324767691 0.6161409753019691 2 2 1]
 [0 2 0.028747611118044934 0.2751445590346463 2 1 1]
 [0 2 0.04094850292506774 0.3962640732534148 2 1 1]
 [2 2 0.5633575299630806 0.5474226843766735 1 2 1]
 [2 2 0.5114209592511445 0.5204278272329121 1 2 1]
 [0 2 0.5114209592511445 0.35012733925919515 1 2 1]
 [0 2 0.5483298686727458 0.4667864684874439 1 2 1]
 [0 2 0.5633575299630806 0.5087047764284351 1 2 1]
 [2 2 0.5114209592511445 0.5105661138440336 2 2 1]
 [0 2 0.028747611118044934 0.32247198473358574 2 1 1]]
Function value: [[-74.64         0.43336017]
 [-78.65         0.74696072]
 [-77.74         0.63557203]
 [-75.22         0.45131919]
 [-76.35         0.48031584]
 [-78.37         0.67800211]
 [-77.19         0.57071544]
 [-76.9          0.53216488]
 [-76.89         0.48936521]
 [-78.51         0.70343561]]
Constraint violation: [[0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]]

'''