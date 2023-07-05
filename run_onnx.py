import numpy as np
import onnx
import onnxruntime
import time

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
#labels = labels[:10000]
#data_noise = data_noise[:10000]
#lr = labels[:,0]

labels = labels[0:4]
lr = labels[0,:]
data_noise = data_noise[0:4]

onnx_model = onnx.load('/project/gruppo1/atlas_gen/rambeluc/DNN_variableTracks/VatTracks.onnx')
sess = onnxruntime.InferenceSession('/project/gruppo1/atlas_gen/rambeluc/DNN_variableTracks/VatTracks.onnx')


print(data_noise.shape)
data_noise = data_noise/np.max(data_noise)
data_noise = data_noise.astype('float32')
#data_noise = np.reshape(data_noise, (data_noise.shape[0],20, 333, 1))
data_noise = np.reshape(data_noise, (4,20,333,1))
batch_size = 4

num_samples = data_noise.shape[0]

timesingle1 = time.time()
output = sess.run(None, {'input_1':data_noise})
timesingle2 = time.time()

print('Single Frame inference time:', (timesingle2-timesingle1))

timesingle1 = time.time()
output = sess.run(None, {'input_1':data_noise})
timesingle2 = time.time()

print('Single Frame inference time:', (timesingle2-timesingle1))

timesingle1 = time.time()
output = sess.run(None, {'input_1':data_noise})
timesingle2 = time.time()

print('Single Frame inference time:', (timesingle2-timesingle1))

timesingle1 = time.time()
output = sess.run(None, {'input_1':data_noise})
timesingle2 = time.time()

print('Single Frame inference time:', (timesingle2-timesingle1))
timesingle1 = time.time()
output = sess.run(None, {'input_1':data_noise})
timesingle2 = time.time()

print('Single Frame inference time:', (timesingle2-timesingle1))
timesingle1 = time.time()
output = sess.run(None, {'input_1':data_noise})
timesingle2 = time.time()

print('Single Frame inference time:', (timesingle2-timesingle1))
timesingle1 = time.time()
output = sess.run(None, {'input_1':data_noise})
timesingle2 = time.time()

print('Single Frame inference time:', (timesingle2-timesingle1))
timesingle1 = time.time()
output = sess.run(None, {'input_1':data_noise})
timesingle2 = time.time()

print('Single Frame inference time:', (timesingle2-timesingle1))
timesingle1 = time.time()
output = sess.run(None, {'input_1':data_noise})
timesingle2 = time.time()

print('Single Frame inference time:', (timesingle2-timesingle1))

timesingle1 = time.time()
output = sess.run(None, {'input_1':data_noise})
timesingle2 = time.time()

print('Single Frame inference time:', (timesingle2-timesingle1))


time1 = time.time()
for i in range(0, num_samples, batch_size):
    input_data = data_noise[i:i+batch_size]
    #print(input_data.shape)
    output = sess.run(None, {"input_1":input_data})

time2 = time.time()

time_total = time2 - time1
fps = float(data_noise.shape[0] / time_total)
print(f"Throughput = {fps} fps, total frames = {data_noise.shape[0]}, time = {time_total} seconds")
