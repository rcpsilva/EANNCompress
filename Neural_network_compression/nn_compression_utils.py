import tensorflow as tf
import numpy as np
import gc
from keras import backend as K

from . import neural_network_utils as nnUtils
from . import pruner
from . import quantizer

def evaluate_tflite_model(tflite_model, test_data):
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  
  print(f'input_index: {input_index}')
  print(f'output_index: {output_index}')

  # Run predictions on ever y image in the "test" dataset.
  predictions = []
  for test_image in test_data[0]:

    image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, image)


    interpreter.invoke()

    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    predictions.append(digit)

  print('\n')
  
  predictions = np.array(predictions)
  total = 0
  for i in range(len(predictions)):
    if (predictions[i] == test_data[1][i]):
      total = total+1
  accuracy = total/len(predictions)
  return accuracy

def eval_solution(X, model, model_layers,data):
  compressed_model = execute_compression(base_model = model, X = X, model_layers = model_layers, training=data[0], validation=data[1])
  if (compressed_model == -1):
    return 0, 9999999999999
  acc = evaluate_tflite_model(tflite_model = compressed_model,test_data= data[2])
  size = nnUtils.get_size_tflite(compressed_model)
  K.clear_session()
  del compressed_model
  gc.collect()
  return acc, size

def execute_compression(base_model, X, model_layers,training, validation):
  if(X[0]):
    base_model = pruner.model_pruner(model = base_model,
                               layers_to_prune = X[2],
                               schedule = X[5], 
                               sparsity = X[3], 
                               frequency = X[6], 
                               convert_to_tflite = not X[1],
                               training = training, 
                               validation = validation,
                              model_layers = model_layers)

  if(X[1]):
    base_model = quantizer.quantization(model = base_model,
                              type_quantization = X[4])
  
  if(not X[1] and not X[0]):
    return -1
  return base_model