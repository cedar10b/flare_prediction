# -*- coding: utf-8 -*-


import time
import numpy as np
from functions import *
from sklearn import svm

t1 = time.clock()

#---------------------------------------------------------------
# START PRE-PROCESSING
print 'Start preprocessing...'
#---------------------------------------------------------------

#define length of time sequence, number of observations, 
#number of features (attributes)
Nt = 61; Nfl = 254; Nnofl=4191; Nattr = 25

#time frame in hours up to which data can be used 
#to make predictions at t=Nt-1
t = 36

#read SHARP data
flare = read_data('flare_catalog.txt', Nt, Nfl, Nattr)
noflare = read_data('noflare_catalog.txt', Nt, Nnofl, Nattr)

#read and normalize flaring history data
flare_hist, noflare_hist = read_normalize_hist('flare_history.txt',
                           'noflare_history.txt',  mode = 'middle')

#clean datacubes from NaNs
flare = clean_nan(flare, mode='constant')
noflare = clean_nan(noflare, mode='constant')

#clean datacubes from missing values (gaps)
flare = clean_gaps(flare, t=t, mode='local_mean')
noflare = clean_gaps(noflare, t=t, mode='local_mean')  
    
#normalize features to have zero mean (or median) and std dev = 1
all_data = np.concatenate((noflare,flare), axis=1)    
flare = normalize(flare, all_data, t=t, mode='median')
noflare = normalize(noflare, all_data, t=t, mode='median')
del all_data   

#clean datacubes from outliers
flare = clean_outlier(flare, lim=5)
noflare = clean_outlier(noflare, lim=5)

#normalize again features, after cleaning outliers
all_data = np.concatenate((noflare,flare), axis=1)
flare = normalize(flare, all_data, t=t, mode='median')
noflare = normalize(noflare, all_data, t=t, mode='median')
del all_data

#-----------------------------------------------------------------
#EXTRACT FEATURES FROM TIME SERIES
print 'Extract features from time series...'
#------------------------------------------------------------------

#define the new statistical features extracted from time series
#possible features are: 
#tval, mean, sdev, ampl, max, min, skew, kurt, slope, 
#t0lin, t0poly, hist
#tval is the value of SHARP features at time t
#t0lin is the ESTIMATED value of SHARP features at flare time 
#(estimated by linear fitting)
#t0poly is the ESTIMATED value of SHARP features at flare time 
#(by 2nd order polynomial)
#slope = the slope of SHARP features determined by linear fitting
#hist = flaring history of the region = log(C*1 + M*10 + X*100)
stat_attr = ['tval', 'mean', 'sdev', 'ampl', 'max', 'min', 
             'slope', 't0lin', 'hist']          

#NnewAttr is the number of new features extracted from time series
if 'hist' in stat_attr:
  NnewAttr = (len(stat_attr)-1)*Nattr + 1   
else:
  NnewAttr = len(stat_attr)*Nattr

#generate new datasets
fl = gen_stat_attr(flare, flare_hist,
                   stat_attr=stat_attr, t=t, Nt=Nt)
nofl = gen_stat_attr(noflare, noflare_hist, 
                     stat_attr=stat_attr, t=t, Nt=Nt)

#labels (1=flare, 0=no flare)
yfl = np.ones(Nfl); ynofl = np.zeros(Nnofl); 

#delete original datacubes to free some memory
del flare; del noflare

#---------------------------------------------------------------
#FEATURE SELECTION
print 'Feature selection...'
#---------------------------------------------------------------

#feature selection. keyword method can be: 
#'Fscore', 'RandomForest', RFE, chi2, pca, DecisionTree
fl, nofl, scores = feature_selection(fl, nofl, 
                   method='Fscore', N_features=75)


#----------------------------------------------------------------
#TRAINING AND TESTING
print 'Training and testing...'
#----------------------------------------------------------------

# number of runs to estimate the error of the measurement
runs = 1000

# SVM algorithm
C = 0.216; gamma = 0.06; class_weight = {1:16.5}
clf = svm.SVC(C=C, gamma=gamma, kernel='rbf', 
              class_weight=class_weight, cache_size=500, 
              max_iter=-1, shrinking=True, tol=1e-8)

# define metrics to measure success
TSS = np.zeros(runs); HSS2 = np.zeros(runs); acc = np.zeros(runs)
TP = np.zeros(runs); TN = np.zeros(runs) 
FP = np.zeros(runs); FN = np.zeros(runs)


for i in range(runs):
  
  #devide datasets into training and test sets 
  #keeping N/P ratio to 16.5 for each of them
  train, test, ytrain, ytest = split_dataset(fl, nofl,
                               test_size=0.3, mode='random')          
  
  # model training and testing
  clf.fit(train, ytrain)
  pred = clf.predict(test)
  TP[i], TN[i], FP[i], FN[i] = confusion_table(pred, ytest)
  
  # compute skill scores
  TSS[i], HSS2[i], acc[i] = \
  compute_scores(TP=TP[i] ,TN=TN[i], FP=FP[i], FN=FN[i], 
                 metrics=['TSS', 'HSS2', 'acc'])        

t2 = time.clock()

# print the mean values of skill scores as well as their st dev
print_results(t=t2-t1, TSS=TSS, HSS2=HSS2, acc=acc)
    
 
 