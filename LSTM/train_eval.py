#Training, and then evaluation of model on train_2.csv dataset

#NOTE: Warning generated for using TensorFlow 1.4 with Python3.6 should be ignored.
#/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/importlib/_bootstrap.py:205:
# RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util'
#does not match runtime version 3.6  return f(*args, **kwds)

from __future__ import print_function, division
import os
import os.path
import numpy as np
import sys, getopt
import math
import csv
import pickle
import datetime
import time
import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt
from matplotlib import gridspec

############ Parameters to control run #############
#Use CPU only or Nvidia GPU
use_cpu_only = False

#Use a fixed initial seed
deterministic_init = True

#Batch size
BS = 1024

#Number of timesteps to look backwards - Back prop length
BL = 36

#Number of steps to look ahead
#FL = 64


#LSTM statesize
stateSize = 50

#Number of Epochs to Iterate over
num_epochs = 30

#Sequence Length
M = 803

#Validation set size - 64 days
#(i.e., from 7/9/2017 to 9/10/2017 prediction)
V = 64

#Learning rate, decay factor
lrate_initial = 2.0
lrate_end = 0.2
ldecay_rate = (lrate_initial - lrate_end)/num_epochs

#Turn on/off animated display with learning
display_on = False

############ Load up saved data  #############
if use_cpu_only:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

if deterministic_init:   
    tf.set_random_seed(1)

X = np.load("Data/Xdata.npy")
Y = np.load("Data/Ydata.npy")

with open("Data/index2Site.pkl", 'rb') as f:
    index2Site = pickle.load(f)
with open("Data/column2Date.pkl", 'rb') as f:
    column2Date = pickle.load(f)
print("Completed loading X, Y and key data ", X.shape, Y.shape)

(nSites, nDates, nFeatures) = X.shape

#Expect to train multiple sites at a time
tstId = range(nSites)

siteName = []

for i in tstId:
    siteName.append(index2Site[i])
    
### Inject synthetic data in X for testing
synthetic_data = False

#If synthetic_data is true, then testing is done with small synthetic subset
if synthetic_data:
    print("Reassigning to synthetic")
    D = np.zeros(nDates, dtype=np.float64)
    M = nDates
    for i in range(M):
        D[i] = (i - M/2.0)/10

    granularity = 14
    D = np.sin((np.array(range(M)) % granularity)*np.pi*2/granularity) +\
            np.sin((np.array(range(M)) % (granularity*2))*np.pi*2/(granularity*2)) +\
            np.sin((np.array(range(M)) % (granularity*3))*np.pi*2/(granularity*3))
    D = D/10 + np.array(range(M))/(M/2)
    tstId = 0
    X = X[0:1,:,0:1]
    X[tstId,:,0] = D
    Y[tstId,:] = D
    #X[tstId,:,1:] = 0
    nFeatures=1

def smapescore(ypred, ytrue):
    y_out = np.expm1(ypred)
    y_actual = np.expm1(ytrue)
    denom = (np.abs(y_actual) + np.abs(y_out)) 
    difference = np.abs(y_actual - y_out) / (2.0 * denom)
    difference[denom == 0] = 0.0
    return np.nanmean(difference)

#Create the LSTM network
X_placeholder = tf.placeholder(tf.float64, [BS,BL*nFeatures]) 
Y_placeholder = tf.placeholder(tf.float64)

cell_state = tf.placeholder(tf.float64, [BS, stateSize])
hidden_state = tf.placeholder(tf.float64, [BS, stateSize])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

lstm = rnn.BasicLSTMCell(stateSize, state_is_tuple=True)

output = []
state = init_state
for i in range(BL):
    output, state = lstm(X_placeholder[0:BS,i*nFeatures:(i+1)*nFeatures], state)
final_state = output


num_classes = 1

W = tf.Variable(np.ones((stateSize, num_classes),dtype=np.float64))
b = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float64)
Ypred = [tf.matmul(final_state, W) + b]

#Mean squared error between predicted and actual sequence
loss = tf.reduce_sum(tf.pow(Ypred - Y_placeholder, 2.0))

# Call optimizer
learning_rate = lrate_initial
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

def gplot2(loss_list, Yactual, Ypred, epoch, M, V):
    gs = gridspec.GridSpec(4, 1)
    segs = 20
    #plt.subplot(2, 1, 1)
    plt.subplot(gs[0,:])
    plt.cla()
    if V == -1:
        plt.title("Loss Values Epoch %3d  (Total Epoch Loss = %0.4e)" % (epoch, sum(loss_list)))
    else:
        plt.title("Prediction for %s\nBL=%d State Size=%d Epochs = %d Smape Score = %0.4e" % (siteName, BL, stateSize, num_epochs, smapescore(Ypred[-V:], Yactual[-V:])))
    Xaxis = range(BL, len(loss_list)+BL, 1)
    plt.ylabel("Loss Value")
    plt.plot(Xaxis, loss_list, label="Loss")
    plt.xticks(np.array(range(0,M+segs,M//segs)))

    #plt.subplot(2, 1, 2)
    plt.subplot(gs[1:,:])
    plt.cla()
    Xaxis = range(BL, len(Yactual)+BL, 1)
    actual, = plt.plot(Xaxis, Yactual, label="Y")
    plt.xlabel("Timestep")
    plt.xticks(np.array(range(0,M+segs,M//segs)))
    
    if V == -1:
        train, = plt.plot(Xaxis, Ypred, label="Ytrain")
        plt.ylabel("Y: Actual vs Train")
        plt.legend([actual, train], ["Y", "Ytrain"])
        plt.xticks(np.array(range(0,M+segs,M//segs)))

    else:
        train, = plt.plot(Xaxis[:-V], Ypred[:-V], label="Ytrain")
        test, = plt.plot(Xaxis[-V:-2], Ypred[-V:-2], label="Ytest")
        plt.xticks(np.array(range(0,M+segs,M//segs)))
        plt.ylabel("Y: Actual vs Train vs Test")
        plt.legend([actual, train, test], ["Y", "Ytrain", "Ytest"])

    plt.draw()
    plt.pause(0.00001)

#Log all results in a file with datename
filename = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
fp = open(filename, 'a')
run_start = str(datetime.datetime.now())
start_time = time.time()

startId = 0
endId = nSites

fp.write("#Run started at %s\n" % run_start)
fp.write("#BS = %d BL = %s Statesize=%d TS_start=%d TS_end=%d\n" % (BS, BL, stateSize, startId, endId))

for i in range(startId, endId, BS):
    tstId = range(i,i+BS,1)
    print("Processing Batch ", tstId)
    fp.write("#Processing Batch \n"+str(tstId))

    _current_cell_state = np.zeros((BS,stateSize),dtype=np.float64)
    _current_hidden_state = np.zeros((BS,stateSize),dtype=np.float64)
    _current_state = _current_cell_state, _current_hidden_state

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if display_on:
            plt.ion()
            plt.figure(figsize=(10, 6))
            plt.show()
            
        loss_list = []
    
        print("Training ...")

        for e in range(num_epochs):
            #optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
            #learning_rate -= ldecay_rate

            F1 = Y[tstId,BL:(M-V)]
            F2 = []
            loss_list = []
            t1 = time.time()
            for i in range(M-V-BL):
                Lstm_in = np.zeros((BS,BL*nFeatures))
                for k in tstId:
                    Lstm_in[k-tstId[0]] = X[k,i:(i+BL)].flatten()
                Lstm_out = Y[tstId,i+BL]
    
                _loss, _optimizer, _final_state, _current_state, _output, _Ypred = sess.run(
                    fetches=[loss, optimizer, final_state, state, output, Ypred],
                    feed_dict={
                        X_placeholder: Lstm_in,
                        Y_placeholder: Lstm_out,
                        cell_state: _current_cell_state,
                        hidden_state: _current_hidden_state
                    })
                if(i==0 and i==1):
                    print("Epoch %d Step %d Truth " % (e,i), "Output=",_output,"Stateout=",_current_state,"Input=",Lstm_in,"Statein=",(_current_cell_state,_current_hidden_state))
                _current_cell_state, _current_hidden_state = _current_state
                loss_list.append(_loss)
                F2.append(float(_Ypred[0][0]))
            if display_on:
                gplot2(loss_list, F1[0], F2, e, M, -1)
            t2 = time.time()
            if(e==0):
                print("Train time per Epoch = %0.3e secs" % (t2-t1))
                fp.write("#Train time per Epoch = %0.3e secs\n" % (t2-t1))
            print('.', end='', flush=True)
    
        #Compute loss outliers - approx BL entries
        totalLoss = sum(loss_list)
        arrLossList = np.array(loss_list)
        arrLossArgs = np.argsort(arrLossList)
        nonOutlierTotalLoss = sum(arrLossList[arrLossArgs[:-BL]])
        print("Training Done")
        #print("Training Done. Total Epoch Loss = %0.4e, Non Outlier Loss = %0.4e"
        #      % (totalLoss, nonOutlierTotalLoss))
    
        print("Evaluation on test data")
        good_state = _current_state
    
        F1 = Y[tstId, BL:]
        F3 = np.zeros((BS,V),dtype=np.float64)

        t1 = time.time()
        for i in range(M-V-BL,M-BL,1):
            Lstm_in = np.zeros((BS,BL*nFeatures))
            for k in tstId:
                Lstm_in[k-tstId[0]] = X[k,i:(i+BL)].flatten()
            Lstm_out = Y[tstId,i+BL]
            _final_state, _current_state, _Ypred = sess.run(
                fetches=[final_state, state, Ypred],
                feed_dict={
                    X_placeholder: Lstm_in,
                    Y_placeholder: Lstm_out,
                    cell_state: _current_cell_state,
                    hidden_state: _current_hidden_state
                })
            _current_cell_state, _current_hidden_state = _current_state
            F2.append(float(_Ypred[0][0]))
            F3[:,(i-(M-V-BL)):(i-(M-V-BL)+1)] = _Ypred[0]

        t2 = time.time()
        print("Evauation time = %0.3e secs" % (t2-t1))
        fp.write("#Evauation time = %0.3e secs\n" % (t2-t1))

        if display_on:
            gplot2(loss_list, F1, F2, e, M, V)

        j = 0
        for i in tstId:
            print("Smape Score %s, \t\t\t\t\t %0.4e" %   (index2Site[i], smapescore(F3[j], F1[j,-V:])))
            fp.write("%s, \t\t\t\t\t %0.4e\n" %   (index2Site[i], smapescore(F3[j], F1[j,-V:])))
            j += 1

run_end = str(datetime.datetime.now())
end_time = time.time()

fp.write("#Run ended at %s\n" % run_end)
fp.write("#Total runtime = %0.4e\n" % (end_time - start_time))
print("Run ended at %s" % run_end)
print("Total runtime = %0.4e" % (end_time - start_time))

            
fp.close()
if display_on:
    plt.ioff()
    plt.show()

        

