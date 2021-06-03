import tensorflow_model_optimization as tfmot
import tensorflow as tf
import keras
from keras.models import load_model
import gc
from keras import backend as K
from numpy import asarray

# Global variables
pruners_schedules =[
            tfmot.sparsity.keras.PolynomialDecay,
            tfmot.sparsity.keras.ConstantSparsity,    
]
layers_to_pruneG = []
sheduleG = 0
sparsityG = []
frequencyG = 0
index_layerG = 0


def variable_to_layers(model_layers, X3):
   layers_to_prune=[]
   for index,x in enumerate(X3):
     if (x):
       layers_to_prune.append(model_layers[index])
   return layers_to_prune

def layers_type_sparsity(model_layers,sparsity):
  layer_sparsity={};
  for index,s in enumerate(sparsity):
    layer_sparsity[model_layers[index]] = s
  return layer_sparsity

def apply_pruning_constant(layer):
  if (layer.__class__.__name__ in layers_to_pruneG):
    new_pruning_params = {
        'pruning_schedule': pruners_schedules[sheduleG](target_sparsity = sparsityG[layer.__class__.__name__], begin_step = 0, end_step=-1, frequency = frequencyG)
    }
    return tfmot.sparsity.keras.prune_low_magnitude(layer,**new_pruning_params)
  return layer

def apply_pruning_polynomial(layer):
  if (layer.__class__.__name__ in layers_to_pruneG):
    new_pruning_params = {
        'pruning_schedule': pruners_schedules[sheduleG](initial_sparsity = sparsityG[layer.__class__.__name__]*0.2 , final_sparsity = sparsityG[layer.__class__.__name__], begin_step = 0, end_step=100, frequency = frequencyG)
    }
    return tfmot.sparsity.keras.prune_low_magnitude(layer,**new_pruning_params)
  return layer

def model_pruner(model, layers_to_prune, schedule, sparsity,frequency, convert_to_tflite, training, validation, model_layers, log_path='./'):
  # setando variaveis globais para serem usadas na função apply_pruning
  #1 - Constant Schecudule
  global index_layerG
  index_layerG = 0
  global layers_to_pruneG
  layers_to_pruneG = variable_to_layers(model_layers=model_layers, X3 = layers_to_prune)
  global sheduleG
  sheduleG = schedule
  global sparsityG
  sparsityG = layers_type_sparsity(model_layers, sparsity)#sparsity
  global frequencyG
  frequencyG = frequency
  if (schedule == 1):
    model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function = apply_pruning_constant,
    )
  else:
    model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function = apply_pruning_polynomial,
    )
  
  log_dir = log_path
  callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
  ]

  model_for_pruning.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
  )

  for i in range(200):
    print(f'fit:{i}')
    model_for_pruning.fit(
        x = asarray(training[0]),
        y = training[1],
        epochs=10,
        validation_data=(asarray(validation[0]),validation[1]),
        batch_size=32,
        callbacks=callbacks 
    )

 # %tensorboard --logdir={log_dir}
  model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
  K.clear_session()
  del model_for_pruning
  gc.collect()
  if (convert_to_tflite):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()
    return pruned_tflite_model

  return model_for_export