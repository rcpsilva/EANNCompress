import tempfile
import os
import zipfile
import tempfile
import shutil
from os import listdir
from os.path import isfile, join



def layers_types(model):
  layers = [] 
  for i in range(1,len(model.layers)):
    type_of_layer =  model.get_layer(index = i).__class__.__name__
    if (not type_of_layer in  layers):
      layers.append(type_of_layer)
  return layers

def get_gzipped_model_file_size(file):
  # Returns size of gzipped model, in bytes.
  size = 0
  try:
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
      f.write(file)
    print('passou aqui')
    size = os.path.getsize(zipped_file)
    print(size)
  except Exception as e:
    print(e)
  finally:
    os.remove(zipped_file)
    return size

def get_size_tflite(model, path = './'):
  _, model_tflite_temp = tempfile.mkstemp(".tflite")
  with open(model_tflite_temp, 'wb') as f:
    f.write(model)
  return get_gzipped_model_file_size(model_tflite_temp)

def get_keras_model_size(model):
  tmp_path = tempfile.mkdtemp()
  model.save(tmp_path)
  tmp_files = [f for f in listdir(tmp_path) if isfile(join(tmp_path, f))]
  tmp_model = tmp_files[0]
  tmp_model_path = tmp_path + '/' + tmp_model
  size = os.stat(tmp_model_path).st_size
  # size = get_gzipped_model_file_size(tmp_model_path)
  shutil.rmtree(tmp_path)
  return size
    