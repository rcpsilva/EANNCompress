import compress_problem
import numpy as np
import pandas as pd
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import tensorflow as tf
from keras.models import load_model
import gc
from tensorflow.keras.datasets import cifar10
from skimage.transform import resize
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
import random
import resnet50_cifar10_training


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

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from skimage.transform import resize
from IPython import embed
import random
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
print('gpus')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print("Error: " + e)


print('1 ------------------------')
NUM_CLASSES = 10
BATCH_SIZE = 24
NUM_EPOCHS = 3
use_data_aug = True

# img_arr is of shape (n, h, w, c)
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


(x_train1, y_train1), (x_test1, y_test1) = cifar10.load_data()

samples = random.sample(range(0,50000), 15000)
samplesV = random.sample(range(0,10000), 1000)

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

algorithm = NSGA2(
  pop_size=200,
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

