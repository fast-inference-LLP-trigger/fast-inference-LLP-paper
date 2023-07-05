import argparse
import os
import shutil
import sys
from sklearn.model_selection import train_test_split
import numpy as np
import time 

import tensorflow as tf
from tensorflow import keras
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


def quant_model(float_model,quant_model,batchsize,evaluate):
    
    tf.test.gpu_device_name()

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

    data_noise = data_noise.astype('float32')

    train_to_test_ratio = 0.8
    X_train,X_test,Y_train,Y_test = train_test_split(data_noise,lr,train_size=train_to_test_ratio, shuffle=True, random_state=1234)


    X_train = X_train.reshape((X_train.shape[0], 20, 333,1))

    float_model = load_model('float_VarTracks.h5')

    quant_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    quant_dataset = quant_dataset.shuffle(X_train.shape[0]).batch(batchsize)

    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=quant_dataset)

    quantized_model.summary()
    quantized_model.save('quant_VarTracks.h5')


    if (evaluate):
       
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        test_dataset = test_dataset.shuffle(X_test.shape[0]).batch(batchsize)

        quantized_model.compile(optimizer=Adam(),
                                loss='mse',
                                metrics=['mae'])

        scores = quantized_model.evaluate(test_dataset,
                                          verbose=1)
        float_model = load_model('float_VarTracks.h5')

       
        start = time.time()
        predictions = quantized_model.predict(data_noise, batch_size = 4)
        end = time.time()

        time_total = end - start
        sample_size = data_noise.shape[0]

        fps = sample_size/time_total

        print(f" Throughput: {fps} fps, Sample size: {sample_size} frames, Total time = {time_total} s")
        
        scores_float = float_model.evaluate(test_dataset, verbose=1)
        float_predictions = float_model.predict(X_train)
        quant_predictions = quantized_model.predict(X_train)
        truth = Y_train
        
        print('Float model mae: ',scores_float[1])
        print('Quantized model mae: ',scores[1])
        
        np.savez('predicted_float_quant', truth, float_predictions, quant_predictions, truth = truth, float_predictions = float_predictions, quant_predictions = quant_predictions)
    
    return



def main():


    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--float_model',  type=str, default='float_10.2.h5', help='Full path of floating-point model. Default is float_model/float_10.h5')
    ap.add_argument('-q', '--quant_model',  type=str, default='quant_model2/quant_10.2.h5', help='Full path of quantized model. Default is quant_model2/q_model.h5')
    ap.add_argument('-b', '--batchsize',    type=int, default=64,                       help='Batchsize for quantization. Default is 64')
    ap.add_argument('-tfdir', '--tfrec_dir',type=str, default='tfrecords',              help='Full path to folder containing TFRecord files. Default is tfrecords')
    ap.add_argument('-e', '--evaluate',     action='store_true', help='Evaluate floating-point model if set. Default is no evaluation.')
    args = ap.parse_args()

    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --float_model  : ', args.float_model)
    print (' --quant_model  : ', args.quant_model)
    print (' --batchsize    : ', args.batchsize)
    print (' --evaluate     : ', args.evaluate)
    print('------------------------------------\n')


    quant_model(args.float_model, args.quant_model, args.batchsize, args.evaluate)


if __name__ ==  "__main__":
    main()
