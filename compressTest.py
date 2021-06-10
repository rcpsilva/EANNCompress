import compress_problem
import numpy as np
import pandas as pd
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import tensorflow as tf
from keras.models import load_model
import gc
from keras import backend as K
from sklearn.model_selection import train_test_split
from pymoo.model.problem import Problem
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from Neural_network_compression import neural_network_utils as nnUtils
from Neural_network_compression import nn_compression_utils as compress
import test_model

base_model = load_model('final_model.h5')
trainX, trainY, testX, testY = test_model.load_dataset()
# prepare pixel data
trainX, testX = test_model.prep_pixels(trainX, testX)

trainX, valX , trainY, valY = train_test_split(trainX, trainY, test_size=0.2)

data= [trainX, trainY], [valX, valY], [testX, testY]

problem = compress_problem.NNCompressProblem(base_model, data)

mask = problem.get_compression_mask()


sampling = MixedVariableSampling(mask, {
    "real": get_sampling("real_random"),
    "int": get_sampling("int_random")
})

crossover = MixedVariableCrossover(mask, {
    "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
})

mutation = MixedVariableMutation(mask, {
    "real": get_mutation("real_pm", eta=3.0),
    "int": get_mutation("int_pm", eta=3.0)
})

algorithm = NSGA2(
  pop_size=10,
  sampling=sampling,
  crossover=crossover,
  mutation=mutation,
  eliminate_duplicates=True,
)
res = minimize(
  problem,
  algorithm,
  ('n_eval', 20),
  seed=69,
  pf=None,
  verbose=True,
  save_history=True
)

