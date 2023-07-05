import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import numpy as np
import tensorflow 
from tensorflow import keras 

inputs = keras.Input(shape=(20,333,1))

x = keras.layers.Conv2D(64, kernel_size=(3,3), padding='same')(inputs)
x = keras.layers.Conv2D(64, kernel_size=(3,3),  padding='same')(x)
x = keras.layers.Conv2D(64, kernel_size=(3,3),  padding='same')(x)
x = keras.layers.ReLU(name='ReLU')(x)
x = keras.layers.MaxPool2D((2,3))(x)

x = keras.layers.Conv2D(128, kernel_size=(3,3), padding='same')(x)
x = keras.layers.ReLU(name='ReLU1')(x)
x = keras.layers.MaxPool2D((2,3))(x)

x = keras.layers.Conv2D(256, kernel_size=(3,3))(x)
x = keras.layers.ReLU(name='ReLU2')(x)
x = keras.layers.MaxPool2D((1,3))(x)

x = keras.layers.Conv2D(256, kernel_size=(3,3))(x)
x = keras.layers.ReLU(name='ReLU3')(x)
x = keras.layers.MaxPool2D((1,3))(x)

x = keras.layers.Flatten()(x)

x = keras.layers.Dense(1024)(x)
x = keras.layers.ReLU(name='ReLU4')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(1024)(x)
x = keras.layers.ReLU(name='ReLU5')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(1)(x)
outputs = keras.layers.ReLU(name='ReLU6')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
