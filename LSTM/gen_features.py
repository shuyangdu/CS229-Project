#Generate data from model

from __future__ import print_function, division
import os
import os.path
import numpy as np
import sys, getopt
import math
import csv
import pickle
import datetime
import re
import matplotlib.pyplot as plt
from matplotlib import gridspec

trainFileName = "Data/train_2.csv"

def loadObjects3(trainFileName, baseName):

    trainSaveFileName = trainFileName + ".npy"
    site2IndexSaveFileName = "Data/" + baseName + "site2Index.pkl"
    index2SiteSaveFileName = "Data/" + baseName + "index2Site.pkl"
    column2DateSaveFileName = "Data/" + baseName + "column2Date.pkl"

    rawTrainArr = np.load(trainSaveFileName)
    with open(site2IndexSaveFileName, 'rb') as f:
        site2Index = pickle.load(f)
    with open(index2SiteSaveFileName, 'rb') as f:
        index2Site = pickle.load(f)
    with open(column2DateSaveFileName, 'rb') as f:
        column2Date = pickle.load(f)

    return (rawTrainArr, site2Index, index2Site, column2Date)

def normalize(x):
    return(math.log(1+x))

vnormalize = np.vectorize(normalize)


def annualAutoCorr(series):
    curr = series[365:]
    mcurr = np.mean(curr)
    prev = series[:-365]
    mprev = np.mean(prev)
    diffc = curr-mcurr
    diffp = prev-mprev
    divider = np.sqrt(np.sum(diffc * diffc)) * np.sqrt(np.sum(diffp * diffp))
    if divider == 0:
        return 0.0
    else:
        return np.sum(diffc * diffp)/divider

def quartAutoCorr(series):
    quarter = 365//4
    curr = series[quarter:]
    mcurr = np.mean(curr)
    prev = series[:-quarter]
    mprev = np.mean(prev)
    diffc = curr-mcurr
    diffp = prev-mprev
    divider = np.sqrt(np.sum(diffc * diffc)) * np.sqrt(np.sum(diffp * diffp))
    if divider == 0:
        return 0.0
    else:
        return np.sum(diffc * diffp)/divider

def monthAutoCorr(series):
    curr = series[30:]
    mcurr = np.mean(curr)
    prev = series[:-30]
    mprev = np.mean(prev)
    diffc = curr-mcurr
    diffp = prev-mprev
    divider = np.sqrt(np.sum(diffc * diffc)) * np.sqrt(np.sum(diffp * diffp))
    if divider == 0:
        return 0.0
    else:
        return np.sum(diffc * diffp)/divider



def date2day(dtstr):
    year, month, day = (int(x) for x in dtstr.split('-'))
    ans = datetime.date(year, month, day)
    #print(ans.strftime("%A"))
    #print(ans.isoweekday())
    ohday = np.zeros(7)
    dnum = ans.isoweekday()-1
    ohday[dnum] = 1
    return ohday


loadObj = loadObjects3(trainFileName, "")
(rawTrainArr, site2Index, index2Site, column2Date) = loadObj

print("Completed loading training and key data\n")
print("Main Train Dim=", rawTrainArr.shape)


# Normalize and create a feature dataset

# Total feats = 1(value) + 7(days of week) + 2(quart, annual autocorr)
# Log(hit_count+1) - feat 0
# Day of week - feat 1 to 7 - one hot
# Monthly Autocorrel - feat 8
# Quarterly Autocorrel - feat 9
# Annual Autocorrel - feat 10
# Traffic median - feat 11
# Country - feat 12-18 
# Access type - feat 19-22

nFeatures = 23
(nSites,nDates) = rawTrainArr.shape

#The expected output is the (1+log) of the raw data
Y = vnormalize(rawTrainArr)
print("Done vnormalize train arr")

#Create an array of the input data
X = np.zeros((nSites, nDates, nFeatures), dtype=np.float32)

#Normalize page hits
X[:,:,0] = Y[:nSites,:nDates]

#Create feats 0, 8-11
for i in range(nSites):
    X[i,:,0] = (X[i,:,0] - X[i,:,0].mean())/X[i,:,0].var()
    X[i,:,8] = monthAutoCorr(X[i,:,0])
    X[i,:,9] = quartAutoCorr(X[i,:,0])
    X[i,:,10] = annualAutoCorr(X[i,:,0])
    X[i,:,11] = np.median(X[i,:,0])
    if(i%20000 == 0):
        print('.', end='', flush=True)
print("\n Done normalizing raw data, corrs, median")
    
#Normalize one-hot week days
#precomputed values == -0.40825, 2.4495 for 0-mean, 1-var numbers
X[:,:,1:8] = -0.40825
X[:,range(0,nDates,7),1] = 2.4495
X[:,range(1,nDates,7),2] = 2.4495
X[:,range(2,nDates,7),3] = 2.4495
X[:,range(3,nDates,7),4] = 2.4495
X[:,range(4,nDates,7),5] = 2.4495
X[:,range(5,nDates,7),6] = 2.4495
X[:,range(6,nDates,7),7] = 2.4495
print("Added day of week")

#Now add country and access type
#Since each is 1-hot, add precomputed 0-mean, 1-var numbers
X[:,:,12:19] = -0.40825
X[:,:,19:23] = -0.5774

preamble = '(.+)_([a-z][a-z]\.)?'
wikipedia_pat = '(?:wikipedia\.org)'
wikimedia_pat = '(?:commons\.wikimedia\.org)'
mediawiki_pat = '(?:www\.mediawiki\.org)'
postamble = '([a-z_-]+?)$'

extractpat = re.compile(preamble + '(' + wikipedia_pat + '|' + 'wikimedia_pat' + '|' + 'mediawiki_pat' + ')_' + postamble)

country2id = {}
cid = 0
agent2id = {}
aid = 0
nmcount = 0
#Create feats for country
#for i in range(145063):
for i in range(145063):
    url = index2Site[i]
    match = extractpat.fullmatch(url)
    if(match == None):
        #print("match = None for url %s" % (url))
        nmcount += 1
        country = "en."
        if(url[-1] == 'r'):
            agent = "all-access_spider"
        elif (url[-15] == "-"):
            agent = 'mobile-web_all-agents'
        elif (url[-12] == "d"):
            agent = 'desktop_all-agents'
        else:
            agent = 'all-access_all-agents'
        #if(i==30000):
            #print("Inferred url %s as (%s,%s)" % (url, country, agent))
    else:
        country = match.group(2)        
        agent = match.group(4)

    if not(country in country2id):
        country2id[country] = cid
        cid += 1
        
    if not(agent in agent2id):
        agent2id[agent] = aid
        aid += 1

    X[i,:,12+country2id[country]] = 2.4495 #Updated one-hot country
    X[i,:,19+agent2id[agent]] = 1.7321 #Updated one-hot access agent

#To Do Add Historical Datapoints Features - quarterly and annual basis

#Save the extracted features and expected output 
np.save("Data/Xdata.npy",X)
np.save("Data/Ydata.npy",Y)


