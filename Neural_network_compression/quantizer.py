import tensorflow as tf

def quantization(model,type_quantization, train_images):
  #convertendo modelo base 
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  # if(type_quantization == 0):
  #   print('convertido pra 16 bits----------------------')
  #   converter.target_spec.supported_types = [tf.float16]
  converter.target_spec.supported_types = [tf.float16]
  tflite_model = converter.convert()
  return tflite_model