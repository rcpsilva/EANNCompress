import numpy as np
import pandas as pd
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import tensorflow as tf
import keras
from keras.models import load_model
import gc
from keras import backend as K

from Neural_network_compression import neural_network_utils as nnUtils
from Neural_network_compression import nn_compression_utils as compress

class NNCompressProblem(Problem):

    def __init__(self, base_model, data, G=[]):
        self.base_model = base_model
        self.data = data
        self.DF =  pd.DataFrame(columns=["Poda","Quantização","Camadas_Podadas","Esparsidade","Tipo_Quantização","Agenda_de_Poda","Frequencia_de_Poda","Acuracia","Tamanho"])
        self.model_layers = nnUtils.layers_types(self.base_model)
        self.base_model_size = nnUtils.get_gzipped_model_file_size(base_model)
        self.num_type_layers = len(self.model_layers)
        num_unitary_problem_variable = 5
        number_variables = num_unitary_problem_variable + self.num_type_layers +self.num_layers 
        xl = np.zeros(number_variables) # bottom limit
        xu = np.ones(number_variables) # upper limit
        xl[-1] = 1 # bottom limit of pruning frequency
        xu[-1] = 100 # upper limit of pruning frequency
        self.G = G
        super().__init__(n_var=number_variables,
                         n_obj=2,
                         n_constr=0,
                         xl=xl,
                         xu=xu,
                         elementwise_evaluation=True)


    def _evaluate(self, x, out, *args, **kwargs):
        variable_vector = self.transform_variable_vector(X = x, num_type_layers = self.num_type_layers ,num_layers = self.num_layers)
        base_model = self.base_model
        acc,size = compress.eval_solution(X=variable_vector, model = base_model, model_layers= self.model_layers, data = self.data)
        K.clear_session()
        del base_model
        gc.collect()
        norm_size = size/self.model_size
        out["F"] = [acc*-1, norm_size]
        out["G"] = self.G

    def transform_variable_vector(X,num_type_layers):
        x1 = X[0] #executar poda
        x2 = X[1] #executar quantização 
        x3 = X[2:2+num_type_layers] # camadas a serem podadas
        x4 = X[2+num_type_layers:2+num_type_layers+num_type_layers]
        x5 = X[-3] #tipo de quantização 0-float16 1-8int
        x6 = X[-2]  #tipo de agenda de poda
        x7 = X[-1] #frequencia de poda
        vector=[x1,x2,x3,x4,x5,x6,x7]
        return vector