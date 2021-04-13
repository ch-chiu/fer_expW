import tensorflow as tf
import model as m
import random
from random import randint
import data as d
import time
import copy
from tensorflow.keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.activations import relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl

batch_size = 32

print("Loading Dataset....")
#Load cifar10
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# y_train = tf.keras.utils.to_categorical(y_train, 10)
# y_test = tf.keras.utils.to_categorical(y_test, 10)

#Load FER2013
#x_train, y_train, x_test, y_test = d.FERLoad()

#Load ExpW
f = open("expW_acc.txt", "a")
x_train, y_train, x_test, y_test = d.ExpwLoad('origin', 0.1)
print("Dataset Loaded")

checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model = m.MobileNet(1.0, 2, relu, False, 0.1, Adam, None, None, None, 2)
history = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=2000, batch_size=batch_size, callbacks=[model_checkpoint_callback])

for i in range(len(history)):
    f.write(str(history.history['val_acc'][i])+"\n")
f.close()

from tensorflow import lite
converter = lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open('test.tflite', 'wb').write(tfmodel)