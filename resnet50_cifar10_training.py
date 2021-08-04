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
import renet50
from sklearn.model_selection import train_test_split

NUM_CLASSES = 10
BATCH_SIZE = 24
NUM_EPOCHS = 6

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

def get_model(from_local=True):
  model = 0
  if from_local:
    model = renet50.ResNet50()
  else:
    inputs = tf.keras.layers.Input(shape=(32,32,3))
    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

    base_model = ResNet50(include_top=False, weights='imagenet')(resize)
    print('6 ------------------------')

    # add a global spatial average pooling layer
    x = base_model
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    # and a logistic layer -- 10 classes for CIFAR10
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=inputs, outputs=predictions)

  # initiate RMSprop optimizer
  opt = keras.optimizers.Adam(0.0001),

  # Let's train the model using RMSprop

  model.compile(optimizer=keras.optimizers.Adam(0.0001),
                loss=keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  return model


if __name__ == '__main__':
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

  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  print('2 ------------------------')

  # samples = random.sample(range(0,50000), 20000)
  # samplesV = random.sample(range(0,10000), 2000)

  # x_train = np.array([x_train1[i] for i in samples])
  # y_train = np.array([y_train1[i] for i in samples])
  # x_test = np.array([x_test1[i] for i in samplesV])
  # y_test = np.array([y_test1[i] for i in samplesV])
  # x_train = x_train1[:5]
  # y_train = y_train1[:5]
  # x_test = x_test1[:5]
  # y_test = y_test1[:5]
  x_train = preprocess_image_input(x_train)
  x_test = preprocess_image_input(x_test)

  # Resize image arrays
  # x_train = resize_image_arr(x_train)
  # x_test = resize_image_arr(x_test)

  # print('3 ------------------------')

  # # Convert class vectors to binary class matrices.
  # y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
  # y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
  # print('4 ------------------------')
  # # Normalize the data
  # x_train = x_train.astype('float32')
  # x_test = x_test.astype('float32')
  # x_train /= 255
  # x_test /= 255
  # print('5 ------------------------')

  # inputs = tf.keras.layers.Input(shape=(32,32,3))
  # resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

  # base_model = ResNet50(include_top=False, weights='imagenet')(resize)
  # print('6 ------------------------')

  # # add a global spatial average pooling layer
  # x = base_model
  # x = GlobalAveragePooling2D()(x)
  # # let's add a fully-connected layer
  # x = tf.keras.layers.Flatten()(x)
  # x = tf.keras.layers.Dense(1024, activation="relu")(x)
  # x = tf.keras.layers.Dense(512, activation="relu")(x)
  # # and a logistic layer -- 10 classes for CIFAR10
  # predictions = Dense(NUM_CLASSES, activation='softmax')(x)

  # # this is the model we will train
  # model = Model(inputs=inputs, outputs=predictions)

  # # initiate RMSprop optimizer
  # opt = keras.optimizers.Adam(0.0001),

  # # Let's train the model using RMSprop

  # model.compile(optimizer=keras.optimizers.Adam(0.0001),
  #               loss=keras.losses.sparse_categorical_crossentropy,
  #               metrics=['accuracy'])

  model = get_model(from_local=False)
  
  model.summary()

  history =model.fit(np.asarray(x_train), y_train,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(np.asarray(x_test),y_test),
              shuffle=False)


  model.save_weights('resnet50_cifar10_weights_new.h5')
  

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2,1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  plt.ylim([min(plt.ylim()),1])
  plt.title('Training and Validation Accuracy')

  plt.subplot(2,1,2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  plt.ylim([0,4.0])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.show()