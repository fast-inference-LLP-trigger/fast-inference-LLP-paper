import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

print("TensorFlow version:", tf.__version__)


print(tf.config.list_physical_devices('GPU'))


model = load_model('float_VarTracks.h5', compile=False)

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

data_noise = np.concatenate((data_noise2, data_noise3, data_noise4, data_noise5, data_noise6, data_noise7, data_noise8, data_noise9, data_noise10), axis = 0)
labels = np.concatenate((labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10), axis = 0)

labels = labels[:4]
#labels = labels[0]
#data_noise = data_noise[0]
lr = labels[:,0]

data_noise = data_noise[:4]
print(data_noise.shape)
data_noise = data_noise/np.max(data_noise)
data_noise = data_noise.astype('float32')
#data_noise = np.reshape(data_noise, (1,20,333,1))
data_noise = np.reshape(data_noise, (data_noise.shape[0],20, 333, 1))

batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (1000/time_total))

batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (1000/time_total))

batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (1000/time_total))

batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (1000/time_total))


batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (1000/time_total))
batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (1000/time_total))
batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (time_total))
batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (time_total))
batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (time_total))
batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (time_total))
batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (time_total))
batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (time_total))
batch = 4
time1 = time.time()
predictions = model.predict(data_noise, batch_size = batch)
time2 = time.time()
time_total = time2 - time1
print('FPS:', (time_total))



sample_size = data_noise.shape[0]

fps = sample_size/time_total

#print(f" Throughput: {fps} fps, Sample size: {sample_size} frames, Total time = {time_total} s")