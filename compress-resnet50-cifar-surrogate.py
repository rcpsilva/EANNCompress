import compress_problem
import numpy as np
from pymoo.util.misc import stack
import tensorflow as tf
import gc
from skimage.transform import resize
from keras import backend as K
from sklearn.model_selection import train_test_split
from pymoo.model.problem import Problem
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.factory import get_crossover, get_mutation, get_sampling
from tensorflow.keras.datasets import cifar10
from skimage.transform import resize
import random

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingRegressor

import resnet50_cifar10_training
import surrogate_optimization
import surrogate_selection
import infill_methods
import sampling
import benchmarks

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
print('gpus')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print("Error: " + e)

def resize_image_arr(img_arr):
    x_resized_list = []
    for i in range(img_arr.shape[0]):
        img = img_arr[0]
        resized_img = resize(img, (224, 224))
        x_resized_list.append(resized_img)
    return np.stack(x_resized_list)

def preprocess_image_input(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims

def get_data():
  (x_train1, y_train1), (x_test1, y_test1) = cifar10.load_data()

  samples = random.sample(range(0,50000), 15000)
  samplesV = random.sample(range(0,10000), 1000)

  x_train = np.array([x_train1[i] for i in samples])
  y_train = np.array([y_train1[i] for i in samples])
  x_test = np.array([x_test1[i] for i in samplesV])
  y_test = np.array([y_test1[i] for i in samplesV])

  x_train = preprocess_image_input(x_train)
  x_test = preprocess_image_input(x_test)

  trainX, valX , trainY, valY = train_test_split(x_train, y_train , test_size=0.2)

  data= [trainX, trainY], [valX, valY], [x_test, y_test]

  return data

def get_resnet50_cifar10():
  model = resnet50_cifar10_training.get_model(False)
  model.load_weights('C:/Users/antonio/Documents/projetos/EANNCompress/resnet50_cifar10_weights.h5')
  return model

def get_problem(model, data):
  problem = compress_problem.NNCompressProblem(model, data)

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

  optimizer = NSGA2(
    pop_size=150,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True,
  )

  return problem, optimizer

def run_surrogate_compression():
  # data = get_data()
  # model = get_resnet50_cifar10()

  (x_train1, y_train1), (x_test1, y_test1) = cifar10.load_data()

  samples = random.sample(range(0,50000), 100)
  samplesV = random.sample(range(0,10000), 100)

  x_train = np.array([x_train1[i] for i in samples])
  y_train = np.array([y_train1[i] for i in samples])
  x_test = np.array([x_test1[i] for i in samplesV])
  y_test = np.array([y_test1[i] for i in samplesV])

  x_train = preprocess_image_input(x_train)
  x_test = preprocess_image_input(x_test)

  model = resnet50_cifar10_training.get_model(False)

  model.load_weights('C:/Users/antonio/Documents/projetos/EANNCompress/resnet50_cifar10_weights.h5')

  trainX, valX , trainY, valY = train_test_split(x_train, y_train , test_size=0.2)

  data= [trainX, trainY], [valX, valY], [x_test, y_test]

  problem, optimizer = get_problem(model, data)
  samples = sampling.rand(problem, 30)

  infill_method= infill_methods.distance_search_space

  surrogate_selection_function = surrogate_selection.mse
        
  surrogate_ensemble = [
    LinearRegression(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    SVR(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    ElasticNetCV(),
    LinearSVR(),
    IsolationForest(),
    GradientBoostingRegressor(),
  ]

  termination = get_termination('n_eval', 20)
  res = surrogate_optimization.optimize(problem,optimizer,termination,
      surrogate_ensemble,samples,infill_method,
      surrogate_selection_function,n_infill=2,
      max_samples=120)

if __name__ == '__main__':
  run_surrogate_compression()




