import os
import shutil
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras 

from tensorflow.keras.metrics import mean_squared_error as mse
from tensorflow.keras.metrics import mean_absolute_error as mae
from tensorflow.keras.regularizers import l1

from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches

from CNN_model import model

fil2 = np.load('dataset/MS_2.npz')
labels2 = fil2['labels']
data_noise2 = fil2['data_noise']

fil3 = np.load('dataset/MS_3.npz')
labels3 = fil3['labels']
data_noise3 = fil3['data_noise']

fil4 = np.load('dataset/MS_4.npz')
labels4 = fil4['labels']
data_noise4 = fil4['data_noise']

fil5 = np.load('dataset/MS_5.npz')
labels5 = fil5['labels']
data_noise5 = fil5['data_noise']

fil6 = np.load('dataset/MS_6.npz')
labels6 = fil6['labels']
data_noise6 = fil6['data_noise']

fil7 = np.load('dataset/MS_7.npz')
labels7 = fil7['labels']
data_noise7 = fil7['data_noise']

fil8 = np.load('dataset/MS_8.npz')
labels8 = fil8['labels']
data_noise8 = fil8['data_noise']

fil9 = np.load('dataset/MS_9.npz')
labels9 = fil9['labels']
data_noise9 = fil9['data_noise']

fil10 = np.load('dataset/MS_10.npz')
labels10 = fil10['labels']
data_noise10 = fil10['data_noise']

def make_binary(dataset):
    data = dataset.flatten()
    data_binary = [1 if i > 0 else 0 for i in data]
    data_binary = np.reshape(data_binary, (dataset.shape[0], dataset.shape[1], dataset.shape[2]))
    print(data_binary.shape)
    return data_binary

data_binary2 = make_binary(data_noise2)
data_binary3 = make_binary(data_noise3)
data_binary4 = make_binary(data_noise4)
data_binary5 = make_binary(data_noise5)
data_binary6 = make_binary(data_noise6)
data_binary7 = make_binary(data_noise7)
data_binary8 = make_binary(data_noise8)
data_binary9 = make_binary(data_noise9)
data_binary10 = make_binary(data_noise10)
data_noise = np.concatenate((data_binary2, data_binary3, data_binary4, data_binary5, data_binary6, data_binary7, data_binary8, data_binary9, data_binary10), axis = 0)
labels = np.concatenate((labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10), axis = 0)
lr = labels[:,0]


max = np.max(data_noise)
data_noise = data_noise/max
data_noise = data_noise.astype('float32')

train_to_test_ratio = 0.8
X_train,X_test,Y_train,Y_test = train_test_split(data_noise,lr,train_size=train_to_test_ratio, shuffle=True, random_state=1234)

X_train = X_train.reshape((X_train.shape[0], 20, 333,1))

LR_ST=1e-3
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR_ST)

model.compile(optimizer=OPTIMIZER,
              loss='mse',
              metrics=['mae'])

def lr_decay(epoch):
  if epoch < 5:
    return LR_ST
  else:
    return LR_ST * tf.math.exp(0.2 * (5 - epoch))

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_decay)

model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath = 'best',
        monitor='val_mae',
        save_weights_only=False,
        save_best_only=True,
        save_freq='epoch')

history = model.fit(X_train, Y_train, epochs=40, batch_size=64,
                    validation_split=0.2, shuffle=True, verbose=1, callbacks=[lr_scheduler, model_checkpoint])


plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

plt.plot(history.history['mae'], label = 'mae')
plt.plot(history.history['val_mae'], label = 'val_mae')
plt.legend()
plt.show()

model.evaluate(X_test, Y_test, verbose=2)
model.save('float_VarTracks.h5')

predictions = model.predict(X_test.reshape((X_test.shape[0],20,333,1)))

plt.hist(predictions, bins=100)
plt.title('Lr predicted')
plt.show()

plt.scatter(predictions,Y_test)
plt.xlabel("$\hat{L}_r$ [m]")
plt.ylabel("$L_r$ [m]")
plt.plot(Y_test, Y_test ,'k-')

