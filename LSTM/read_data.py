#Read in training and key files, and write out corresponding numpy and pkl files.

import os
import os.path
import numpy as np
import sys, getopt
import math
import csv
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dateutil.parser as dparser

debug = False

trainFileName = "Data/train_2.csv"
keyFileName =   "Data/key_2.csv"


def loadData(trainFileName, keyFileName):
    train = loadTrainData(trainFileName)
    key = loadKeyData(keyFileName)
    return (train, key)

def loadKeyData(keyFileName):
    keyArr = None
    keyTmp = []
    
    with open(keyFileName, newline='') as csvfile:
        keyReader = csv.reader(csvfile)
        firstRow = True
        rowCount = 0

        for row in keyReader:
            try:
                if firstRow:
                    firstRow = False
                else:
                    rowCount = rowCount + 1
                    if debug:
                        print(row)
                        if rowCount == 10:
                            break
                    if(len(row) != 2):
                        raise ValueError('Row length!= 2')
                    keyTmp.append(row)

            except ValueError as err:
                print("Error in ", row, err.args)
                return


    keyArr = np.array(keyTmp, dtype='str')


    if debug:
        print(keyArr)
        print(keyArr.shape)
        print(keyArr.dtype)


    return keyArr


def loadTrainData(trainFileName):
    trainArr = None
    site2index = {}
    index2site = {}
    column2date = []

    tmpArr = []
    with open(trainFileName, newline='') as csvfile:
        trainReader = csv.reader(csvfile)#, delimiter=' ', quotechar='|')
        firstRow = True
        rowCount = 0
        numDates = 0
        for row in trainReader:
            try:
                if firstRow:
                    numDates = len(row)-1
                    column2date = row[1:]
                    firstRow = False
                else:
                    siteName = row[0]                
                    if(row[0][0] == '"' and row[0][-1] == '"'):
                        siteName = row[0][1:-1]
                    site2index[siteName] = rowCount
                    index2site[rowCount] = siteName
                    tmpArr.append(list(map((lambda x: 0 if x=='' else float(x)), row[1:])))
                    rowCount = rowCount + 1
                    if(debug and rowCount == 10):
                        break
            except ValueError as err:
                print("Error in ", row, err.args)
                return


    trainArr = np.array(tmpArr)
    
    if debug:
        print(trainArr.shape)
        print(trainArr.dtype)
        print(trainArr[0:10,0:5])
        print(site2index)
        print(index2site)
        print(column2date)

    return (trainArr, site2index, index2site, column2date)

def loadObjects(trainFileName, keyFileName, baseName):

    trainSaveFileName = trainFileName + ".npy"
    site2IndexSaveFileName = "Data/" + baseName + "site2Index.pkl"
    index2SiteSaveFileName = "Data/" + baseName + "index2Site.pkl"
    column2DateSaveFileName = "Data/" + baseName + "column2Date.pkl"
    keySaveFileName = keyFileName + ".npy"

    trainArr = np.load(trainSaveFileName)
    with open(site2IndexSaveFileName, 'rb') as f:
        site2Index = pickle.load(f)
    with open(index2SiteSaveFileName, 'rb') as f:
        index2Site = pickle.load(f)
    with open(column2DateSaveFileName, 'rb') as f:
        column2Date = pickle.load(f)
    keyArr = np.load(keySaveFileName)

    return (trainArr, site2Index, index2Site, column2Date, keyArr)


def saveObjects(train, key, trainFileName, keyFileName, baseName):

    (trainArr, site2Index, index2Site, column2Date) = train
    keyArr = key
    
    trainSaveFileName = trainFileName + ".npy"
    site2IndexSaveFileName = "Data/" + baseName + "site2Index.pkl"
    index2SiteSaveFileName = "Data/" + baseName + "index2Site.pkl"
    column2DateSaveFileName = "Data/" + baseName + "column2Date.pkl"
    keySaveFileName = keyFileName + ".npy"
    
    np.save(trainSaveFileName, trainArr)
    with open(site2IndexSaveFileName, 'wb') as f:
        pickle.dump(site2Index, f, pickle.HIGHEST_PROTOCOL)
    with open(index2SiteSaveFileName, 'wb') as f:
        pickle.dump(index2Site, f, pickle.HIGHEST_PROTOCOL)
    with open(column2DateSaveFileName, 'wb') as f:
        pickle.dump(column2Date, f, pickle.HIGHEST_PROTOCOL)
    #np.save(keySaveFileName,   keyArr)
   

def main():
    os.system('date')
    (train, key) = loadData(trainFileName, keyFileName)
    print("Loaded Data")
    os.system('date')
    saveObjects(train, key, trainFileName, keyFileName, "")
    print("Saved Data")
    os.system('date')
    #loadObj = loadObjects(trainFileName, keyFileName, "")
    #print("Reloaded Data")
    #os.system('date')
    
if __name__ == '__main__':
    main()
    
