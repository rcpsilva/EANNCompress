import tempfile
import os
import zipfile

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
  except Exception as e:
    print(e)
  finally:
    #os.remove(zipped_file)
    return size

def get_size_tflite(model, path = './'):
  with open(path, 'wb') as f:
    f.write(model)  
  return get_gzipped_model_file_size(path)