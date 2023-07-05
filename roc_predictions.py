import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

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


lr2 = labels2[:,0]
lr3 = labels3[:,0]
lr4 = labels4[:,0]
lr5 = labels5[:,0]
lr6 = labels6[:,0]
lr7 = labels7[:,0]
lr8 = labels8[:,0]
lr9 = labels9[:,0]
lr10 = labels10[:,0]

print(lr2.shape, data_noise2.shape)

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

data_noise2 = np.reshape(data_binary2, (data_noise2.shape[0], 20, 333, 1))
data_noise3 = np.reshape(data_binary3, (data_noise3.shape[0], 20, 333, 1))
data_noise4 = np.reshape(data_binary4, (data_noise4.shape[0], 20, 333, 1))
data_noise5 = np.reshape(data_binary5, (data_noise5.shape[0], 20, 333, 1))
data_noise6 = np.reshape(data_binary6, (data_noise6.shape[0], 20, 333, 1))
data_noise7 = np.reshape(data_binary7, (data_noise7.shape[0], 20, 333, 1))
data_noise8 = np.reshape(data_binary8, (data_noise8.shape[0], 20, 333, 1))
data_noise9 = np.reshape(data_binary9, (data_noise9.shape[0], 20, 333, 1))
data_noise10 = np.reshape(data_binary10, (data_noise10.shape[0], 20, 333, 1))


print(lr2.shape, data_noise2.shape)


model = load_model('float_VarTracks.h5')

predictions2 = model.predict(data_noise2, batch_size = 4)
predictions3 = model.predict(data_noise3, batch_size = 4)
predictions4 = model.predict(data_noise4, batch_size = 4)
predictions5 = model.predict(data_noise5, batch_size = 4)
predictions6 = model.predict(data_noise6, batch_size = 4)
predictions7 = model.predict(data_noise7, batch_size = 4)
predictions8 = model.predict(data_noise8, batch_size = 4)
predictions9 = model.predict(data_noise9, batch_size = 4)
predictions10 = model.predict(data_noise10, batch_size = 4)

print(predictions2.shape)


np.savez('predicted_float_2', lr2, predictions2, lr2 = lr2, predictions2 = predictions2)
np.savez('predicted_float_3', lr3, predictions3, lr3 = lr3, predictions3 = predictions3)
np.savez('predicted_float_4', lr4, predictions4, lr4 = lr4, predictions4 = predictions4)
np.savez('predicted_float_5', lr5, predictions5, lr5 = lr5, predictions5 = predictions5)
np.savez('predicted_float_6', lr6, predictions6, lr6 = lr6, predictions6 = predictions6)
np.savez('predicted_float_7', lr7, predictions7, lr7 = lr7, predictions7 = predictions7)
np.savez('predicted_float_8', lr8, predictions8, lr8 = lr8, predictions8 = predictions8)
np.savez('predicted_float_9', lr9, predictions9, lr9 = lr9, predictions9 = predictions9)
np.savez('predicted_float_10', lr10, predictions10, lr10 = lr10, predictions10 = predictions10)
