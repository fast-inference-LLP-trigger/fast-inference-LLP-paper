from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import json

import shutil
import sys

import vaitrace_py
from vaitrace_py import vai_tracepoint

divider = '---------------------'

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

@vai_tracepoint
def runDPU(id,start,dpu,img):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
 
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]


        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])

       
        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)
        time_dpus2 = time.time()
        '''store output vector'''
        for j in range(runSize):
            out_q[write_index] = outputData[0][j]
            write_index += 1
        count = count + runSize
        

@vai_tracepoint
def app(threads,model):

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

    print(data_noise.shape, labels.shape)
    print(data_noise.shape)
    print(lr.shape)
    
    data_noise = data_noise.astype('float32')

    X_test = data_noise[0:36000]
    Y_test = lr[0:36000]
    X_test = np.reshape(X_test, (len(Y_test),20,333,1))

    img = []
    for i in range(X_test.shape[0]):
        img.append(X_test[i])

    runTotal = len(img) 
   
    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []

    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    '''run threads '''

    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)

        time2_thr1 = time.time()
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end
        time2_thr2 = time.time()
        print('Creation time for thread ', i, ' is ', (time2_thr2-time2_thr1) )

    time1 = time.time()
    for x in threadAll:
        x.start()
       
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print("Throughput=%.4f fps, total frames = %.0f, time=%.8f seconds" %(fps, runTotal, timetotal))

    tpf = float(1/fps)
    mse = 0
    fpga_predictions = []
    fpga_truth = []
    for i in range(len(out_q)):
        prediction = out_q[i][0]
        fpga_predictions.append(prediction)
        ground_truth = Y_test[i]
        fpga_truth.append(ground_truth)
        mse += (prediction - ground_truth)**2
    print(len(fpga_predictions), len(fpga_truth))
    print('mse: ', float(mse/len(out_q)))

   
    print('Time per frame (s): ', tpf)
    print('Time total (s) : ', timetotal)
    np.savez('fpga_predicted_250', fpga_truth, fpga_predictions, fpga_truth = fpga_truth, fpga_predictions = fpga_predictions)
    
    return



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--threads', type = int, default=1, help = 'Number of threads. Default 1.')
    ap.add_argument('-m', '--model', type=str, default='customcnn.xmodel', help='Path of xmodel. Default is customcnn.xmodel')


    args = ap.parse_args()

    print(divider)
    print ('Command line options:')
    print (' --threads   : ', args.threads)
    print (' --model     : ', args.model)

    app(args.threads,args.model)

if __name__ == '__main__':
  main()
