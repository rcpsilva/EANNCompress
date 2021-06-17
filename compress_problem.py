import numpy as np
import pandas as pd
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import tensorflow as tf
from keras.models import load_model
import gc
from keras import backend as K
import os

from Neural_network_compression import neural_network_utils as nnUtils
from Neural_network_compression import nn_compression_utils as compress

class NNCompressProblem(Problem):

    def __init__(self, base_model, data, G=[], fit_params=None):
        self.base_model = base_model
        self.data = data
        self.DF =  pd.DataFrame(columns=["Pruning","Quantization","Pruned_Layers","Sparsity","Quantization_type","Pruning_Schedule","Pruning_Frequency","Accuracy","Model_Size"])
        self.model_layers = nnUtils.layers_types(self.base_model)
        #self.base_model_size = nnUtils.get_gzipped_model_file_size(base_model)
        self.base_model_size = nnUtils.get_keras_model_size(base_model)#os.stat('final_model.h5').st_size
        self.num_type_layers = len(self.model_layers)
        self.fit_params = fit_params
        num_unitary_problem_variable = 5
        number_variables = num_unitary_problem_variable + self.num_type_layers * 2 # both sparsity and pruned layer have the same size that is the number of distinct layer in the model
        xl = np.zeros(number_variables) # bottom limit
        xu = np.ones(number_variables) # upper limit
        xl[-1] = 1 # bottom limit of pruning frequency
        xu[-1] = 200 # upper limit of pruning frequency
        self.G = G
        n_constr = len(G)
        super().__init__(n_var=number_variables,
                         n_obj=2,
                         n_constr=n_constr,
                         xl=xl,
                         xu=xu,
                         elementwise_evaluation=True)


    def _evaluate(self, x, out, *args, **kwargs):
        variable_vector = self.transform_variable_vector(X = x, num_type_layers = self.num_type_layers)
        base_model = tf.keras.models.clone_model(self.base_model)
        base_model.set_weights(self.base_model.get_weights())
        acc,size = compress.eval_solution(X=variable_vector,
            model = base_model,
            model_layers= self.model_layers,
            data = self.data,
            fit_params=self.fit_params)
        K.clear_session()
        del base_model
        gc.collect()
        norm_size = size/self.base_model_size
        out["F"] = [acc*-1, norm_size]
        out["G"] = self.G

    def transform_variable_vector(self, X, num_type_layers):
        x1 = X[0] #executar poda
        x2 = X[1] #executar quantização 
        x3 = X[2:2+num_type_layers] # camadas a serem podadas
        x4 = X[2+num_type_layers:2+num_type_layers+num_type_layers]
        x5 = X[-3] #tipo de quantização 0-float16 1-8int
        x6 = X[-2]  #tipo de agenda de poda
        x7 = X[-1] #frequencia de poda
        vector=[x1,x2,x3,x4,x5,x6,x7]
        return vector

    def get_compression_mask(self):
        num_unitary_problem_variable = 5
        num_type_layers = self.num_type_layers
        number_variables = num_unitary_problem_variable + num_type_layers * 2
        mask=["int"] * number_variables

        for i in range(num_type_layers):
            mask[2+num_type_layers+i]="real"
        return mask

# if __name__ == "__main__":
    
#     problem = MyProblem(base_model_path='', data=data)

#     algorithm = NSGA2(
#         pop_size=10,
#         sampling=sampling,
#         crossover=crossover,
#         mutation=mutation,
#         eliminate_duplicates=True,
#     )

#     res = minimize(
#         problem,
#         algorithm,
#         ('n_eval', 400),
#         seed=69,
#         pf=None,
#         verbose=True,
#         save_history=True
#     )