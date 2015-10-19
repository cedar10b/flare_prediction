# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats
from scipy import random
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFE
from sklearn.decomposition import PCA



def read_data(filename, Nt, Nobs, Nattr):
  """
  reads the data and reshapes them into a 3D numpy array (time, region, feature)
  the flaring time is at the end of the time sequence
  
  Args:
  filename: string, the name of the file (or path + name)
  Nt: int, the number of time frames
  Nobs: int, the number of observations
  Nattr: int the number of features (attributes)
  
  Returns:
  3D numpy array (time, region, feature) with the flaring time at the end of the time sequence
  """
  data = np.loadtxt(filename, dtype=float)
  data = np.reshape(data, (Nt, Nobs, Nattr))
  return data
  


def read_normalize_hist(flare_filename, noflare_filename, mode='median'):
  """
  reads and normalizes the flaring history data for the flaring and nonflaring classes
  history = log(C*1 + M*10 + X*100)
  history needs to be normalized from -1 to 1 (a logistic function is used)
  
  Args:
  flare_filename: string, the name of the flaring history file for flaring regions
  noflare_filename: string, the name of the flaring history file for nonflaring regions
  mode: string, 'mean' or 'median' for standardization  
  
  Returns:
  1D numpy array with flaring history of each selected region
  """
  fl_data = np.loadtxt(flare_filename, dtype=float)
  nofl_data = np.loadtxt(noflare_filename, dtype=float)
  fl_data = np.where(fl_data > 0, np.log(fl_data), 0)
  nofl_data = np.where(nofl_data > 0, np.log(nofl_data), 0)
  data = np.concatenate((fl_data, nofl_data))
  if mode == 'median':
    ave = np.median(data)
  elif mode == 'mean':
    ave = np.mean(data)
  elif mode == 'middle':
   ave = (np.min(data) + np.max(data))/2.
  #Normalize flare history from -1 to 1. 
  Nfl = fl_data.shape[0]; Nnofl = nofl_data.shape[0]
  for i in range(Nfl):
    fl_data[i] = (1. - np.exp(fl_data[i]-ave))/(1. + np.exp(fl_data[i]-ave))
  for i in range(Nnofl):
    nofl_data[i] = (1. - np.exp(nofl_data[i]-ave))/(1. + np.exp(nofl_data[i]-ave))
  return fl_data, nofl_data

  

def clean_nan(data, mode='constant'):
  """
  cleans a dataset from NaNs
  
  Args:
  data: 3D numpy array (time, region, feature) to be cleaned
  mode: string, either 'average' or 'constant', if set to average, nans will be replaced
        with the average of the previous and next frame, if set to constant, nans will be
        replaced with a constant value = -0.010203089565 (bad pixel value)
  
  Returns:
  3D numpy array (time, region, feature) that is cleaned
  """
  nan_loc = np.where(np.isnan(data) == True)
  if mode == 'average':
    for i in range(np.shape(nan_loc)[1]):
      xnan = nan_loc[0][i]; ynan = nan_loc[1][i]; znan = nan_loc[2][i] 
      if (xnan >= 1) and (xnan <= np.shape(data)[0]-2): 
        data[xnan, ynan, znan] = (data[xnan-1,ynan,znan] + data[xnan+1,ynan,znan])/2.
  elif mode == 'constant':
    bad_pix_value = -0.010203089565  
    for i in range(np.shape(nan_loc)[1]):
      xnan = nan_loc[0][i]; ynan = nan_loc[1][i]; znan = nan_loc[2][i]
      data[xnan, ynan, znan] = bad_pix_value
  return data

def clean_gaps(data, t, mode='local_mean'):
  """
  fills gaps with mean value of the same time sequence computed without gaps
  
  Args:
  data: 3D numpy array (time, region, feature) to be cleaned
  t: int, indicates time until which data will be cleaned
  mode: string, 'global_mean' or 'local_mean' 
        global_mean fills gaps with mean value of the whole time sequence
        local_mean fills gaps with the local mean in the area of the gap 
  
  Returns:
  3D numpy array (time, region, feature) that is cleaned
  """
  bad_pix_value = -0.010203089565
  ind = np.where(data[0:t+1,:,:] == bad_pix_value)  
  if mode == 'global_mean':
    for j in range(np.shape(ind)[1]):
      #compute mean of the time sequence until time t excluding gaps  
      mean = np.mean(data[0:t+1,ind[1][j],ind[2][j]] \
                         [data[0:t+1,ind[1][j],ind[2][j]] != bad_pix_value])
      data[ind[0][j],ind[1][j],ind[2][j]] = mean
    return data
  elif mode == 'local_mean':
    width = 10
    for j in range(np.shape(ind)[1]):
      low = max(0, ind[0][j] - width)
      high = min(t, ind[0][j] + width)
      mean=np.mean(data[low:high+1,ind[1][j],ind[2][j]] \
                  [data[low:high+1,ind[1][j],ind[2][j]] != bad_pix_value])          
      data[ind[0][j],ind[1][j],ind[2][j]] = mean        
    return data            

def normalize(data, ref_data, t=36, mode='median'):
  """
  Transforms a datacube into standard format with zero mean and std dev equal to one.
  
  Args:
  data: 3D numpy array (time, region, feature) to be standardized
  ref_data: 3D numpy array (time, region, feature) that is used to compute mean and std dev
            may be same or different than data
  t: int indicating time until which data from ref_data are used to compute mean and std dev
  mode: string, use of either "median" or "mean" for standardization

  Returns:
  3D numpy array (time, region, feature) that is standardized         
  """  
  Nattr = np.shape(data)[2]  
  for i in range(Nattr):
    if mode == 'median':  
      ave = np.median(ref_data[0:t+1,:,i]); sdev = np.std(ref_data[0:t+1,:,i])
    elif mode == 'mean':   
      ave = np.mean(ref_data[0:t+1,:,i]); sdev = np.std(ref_data[0:t+1,:,i]) 
    data[:,:,i] -= ave
    data[:,:,i] /= sdev  
  return data
  
  
def clean_outlier(data, lim=10):
  """
  cleans a datacube from outliers using previous and/or next time frames

  Args:
  data: 3D numpy array (time, region, feature) to be cleaned
  lim: int, the threshold to define an outlier, typically 10-20 std dev jump from one frame to next
  
  Returns:
  3D numpy array (time, region, feature) that is cleaned from outliers
  """    
  Nt=np.shape(data)[0]; Nobs=np.shape(data)[1]; Nattr=np.shape(data)[2]
  fixed_loc = []
  for j in range(Nobs):
    for k in range(Nattr):
      for i in range(1,Nt-1):
        if (abs(data[i+1,j,k]-data[i,j,k])>=lim) and (abs(data[i,j,k]-data[i-1,j,k])>=lim):
          data[i,j,k] = (data[i-1,j,k]+data[i+1,j,k])/2.
          fixed_loc.append((i,j,k))
      for i in range(1,Nt-2):    
        if (abs(data[i,j,k]-data[i-1,j,k])>=lim) and (abs(data[i,j,k]-data[i+1,j,k])<=lim) \
        and (abs(data[i+2,j,k]-data[i+1,j,k])>=lim):
          data[i,j,k] = (data[i-1,j,k] + data[i+2,j,k])/2. 
          data[i+1,j,k] = (data[i-1,j,k] + data[i+2,j,k])/2.
          fixed_loc.append((i,j,k)); fixed_loc.append((i+1,j,k))  
      if (abs(data[0,j,k]) >= lim) and (abs(data[0,j,k]-data[1,j,k]) >= lim) and \
      (abs(data[1,j,k]) <= lim):
        data[0,j,k] = np.mean(data[1:6,j,k])
        fixed_loc.append((0,j,k))
      if (abs(data[0,j,k]) >= lim) and (abs(data[1,j,k]) >= lim) and \
      (abs(data[2,j,k]) <= lim) and (abs(data[2,j,k]-data[1,j,k]) >= lim):
        data[0,j,k] = np.mean(data[2:7,j,k])
        data[1,j,k] = np.mean(data[2:7,j,k])
        fixed_loc.append((0,j,k)); fixed_loc.append((1,j,k))
  return data      
  
def mad_cleaning(data, threshold=20):
  """
  implementation of median absolute deviation from the median for data cleaning
  
  Args:
  data: 3D numpy array (time, region, feature) to be cleaned
  threshold: int, the threshold used by the MAD method
  
  Returns:
  3D numpy array (time, region, feature) that is MAD cleaned from outliers
  """
  Nt=np.shape(data)[0]; Nobs=np.shape(data)[1]; Nattr=np.shape(data)[2]
  width = 8
  fixed_loc = []
  for j in range(Nobs):
    for k in np.delete(np.arange(Nattr),16):
      med = np.median(data[:,j,k])
      ind = np.where(data[:,j,k] > med + threshold*np.median(abs(data[:,j,k] - med)))
      if len(ind[0] > 0):
        for i in range(len(ind[0])):
          fixed_loc.append((ind[0][i],j,k))
          low = np.max([0, ind[0][i] - width])
          high = np.min([Nt, ind[0][i] + width])
          indices = np.arange(low,high)[~np.in1d(np.arange(low,high), ind[0])]
          data[ind[0][i],j,k] = np.mean(data[indices,j,k])
  return data   
      
def gen_stat_attr(data, data_hist, stat_attr, t=36, Nt=61):
  """
  computes predefined statistical features such as mean, max etc 
  that characterize the temporal evolution of SHARP time series   
  
  Args:
  data: 3D numpy array (time, region, feature) 
  data_hist: 1D numpy array with flaring history
  stat_attr: list of strings that determine the statistical features
  t: int, indicates time until which data can be used to compute statistical features
  Nt: int, indicates time of flare in hours relative to beginning of time sequence
  
  Returns:
  2D numpy array (Nobs, NnewAttr)
  """  
  Nt=np.shape(data)[0]; Nobs=np.shape(data)[1]; Nattr=np.shape(data)[2]
  if 'hist' in stat_attr:
    NnewAttr = (len(stat_attr)-1)*Nattr + 1 #number of new features extracted from time series  
  else:
    NnewAttr = len(stat_attr)*Nattr
  dt = np.zeros((Nobs, NnewAttr)) #initialize the new dataset
  attr_counter = 0  
  if 'tval' in stat_attr:  
    for i in range(Nattr):
      dt[:,i] = data[t,:,i]
    attr_counter += Nattr
  if 'mean' in stat_attr:  
    for i in range(Nattr):
      dt[:,attr_counter+i] = np.mean(data[0:t+1,:,i], axis=0)
    attr_counter += Nattr    
  if 'sdev' in stat_attr:  
    for i in range(Nattr):
      dt[:,attr_counter+i] = np.std(data[0:t+1,:,i], axis=0)
    attr_counter += Nattr        
  if 'ampl' in stat_attr:  
    for i in range(Nattr):
      dt[:,attr_counter+i] = np.max(data[0:t+1,:,i], axis=0) - np.min(data[0:t+1,:,i], axis=0)
    attr_counter += Nattr        
  if 'max' in stat_attr:  
    for i in range(Nattr):
      dt[:,attr_counter+i] = np.max(data[0:t+1,:,i], axis=0)
    attr_counter += Nattr    
  if 'min' in stat_attr:  
    for i in range(Nattr):
      dt[:,attr_counter+i] = np.min(data[0:t+1,:,i], axis=0)
    attr_counter += Nattr
  if 'skew' in stat_attr:  
    for i in range(Nattr):
      dt[:,attr_counter+i] = scipy.stats.skew(data[0:t+1,:,i])
    attr_counter += Nattr
  if 'kurt' in stat_attr:  
    for i in range(Nattr):
      dt[:,attr_counter+i] = scipy.stats.kurtosis(data[0:t+1,:,i])
    attr_counter += Nattr
  if 'slope' in stat_attr:
    for i in range(Nattr):
      for j in range(Nobs):
        ydata = data[0:t+1, j, i].reshape((t+1,1))
        xdata = np.arange(t+1).reshape((t+1,1))
        dt[j,attr_counter+i] = LinearRegression().fit(xdata,ydata).coef_
    attr_counter += Nattr
  if 't0lin' in stat_attr:
    for i in range(Nattr):
      for j in range(Nobs):
        ydata = data[0:t+1, j, i].reshape((t+1,1))
        xdata = np.arange(t+1).reshape((t+1,1))
        dt[j,attr_counter+i] = LinearRegression().fit(xdata,ydata).predict(Nt-1)
    attr_counter += Nattr
  if 't0poly' in stat_attr:
    for i in range(Nattr):
      for j in range(Nobs):
        ydata = data[0:t+1, j, i].reshape((t+1,1))
        xdata = np.arange(t+1).reshape((t+1,1))
        xdata = PolynomialFeatures(degree=2).fit_transform(xdata)[:,1:3]
        #dt[j,attr_counter+i] = LinearRegression().fit(xdata,ydata).predict((Nt-1,(Nt-1)**2))
        dt[j,attr_counter+i] = Lasso(alpha=1.0).fit(xdata,ydata).predict((Nt-1,(Nt-1)**2))
    attr_counter += Nattr
  if 'hist' in stat_attr:
    dt[:, attr_counter] = data_hist    
  return dt
 


def feature_selection(fl, nofl, method='Fscore', N_features=10):
  """
  Selects most important features according to F-score, entropy etc
  
  Args:
  fl: 2D np array (region, feature) of flaring regions
  nofl: 2D np array (region, feature) of nonflaring regions
  method: string, choose one of: 'Fscore', 'RandomForest', 'RFE', 'chi2', 'pca', 'DecisionTree'
  N_features: integer, number of features to be selected
  
  Returns:
  fl: 2D transformed array (region, only important features) of flaring regions
  nofl: 2D transformed array of nonflaring regions
  scores: 1D array with size N_features which has the scores of the features (e.g. F score)  
          float for pca that shows variance explained  
  """
  Nfl = fl.shape[0]; Nnofl = nofl.shape[0] 
  yfl = np.ones(Nfl); ynofl = np.zeros(Nnofl)
  if method == 'RandomForest':  
    selector = RandomForestClassifier(n_estimators=10000, criterion='entropy', \
                                      class_weight='auto', max_features = 0.5)
    selector.fit(np.concatenate((fl,nofl),axis=0), np.concatenate((yfl, ynofl), axis=0))
    scores=selector.feature_importances_
    #threshold = sorted(scores, reverse=True)[N_features-1]
    #fl = selector.transform(fl, threshold=threshold) 
    #nofl = selector.transform(nofl, threshold=threshold)
    fl = fl[:, np.argsort(scores)[::-1][0:N_features]]
    nofl = nofl[:, np.argsort(scores)[::-1][0:N_features]]
  elif method == 'DecisionTree':
    selector = DecisionTreeClassifier(criterion='entropy', class_weight='auto')
    selector.fit(np.concatenate((fl,nofl),axis=0), np.concatenate((yfl, ynofl), axis=0))
    scores=selector.feature_importances_
    fl = fl[:, np.argsort(scores)[::-1][0:N_features]]
    nofl = nofl[:, np.argsort(scores)[::-1][0:N_features]]
  elif method == 'RFE':
    estimator = LogisticRegression(penalty='l1', class_weight='auto')
    selector = RFE(estimator, n_features_to_select=N_features, step=1)
    selector = selector.fit(np.concatenate((fl,nofl),axis=0), np.concatenate((yfl, ynofl), axis=0))
    scores = selector.ranking_
    fl = fl[:, np.argsort(scores)[0:N_features]]
    nofl = nofl[:, np.argsort(scores)[0:N_features]]
  elif method == 'Fscore':
    selector = SelectKBest(f_classif, k=N_features)
    selector.fit(np.concatenate((fl,nofl),axis=0), np.concatenate((yfl, ynofl), axis=0))
    scores=selector.scores_
    #fl = selector.transform(fl); nofl = selector.transform(nofl)
    fl = fl[:, np.argsort(scores)[::-1][0:N_features]]
    nofl = nofl[:, np.argsort(scores)[::-1][0:N_features]]
  elif method == 'chi2':
    data = np.concatenate((fl,nofl),axis=0)
    minim = np.zeros(fl.shape[1])
    for i in range(fl.shape[1]):
      minim[i] = np.min(data[:,i])
      if minim[i] < 0:
        fl[:,i] = fl[:,i] - minim[i]; nofl[:,i] = nofl[:,i] - minim[i]  
    selector = SelectKBest(chi2, k=N_features)
    selector.fit(np.concatenate((fl,nofl),axis=0), np.concatenate((yfl, ynofl), axis=0))
    scores=selector.scores_
    #fl = selector.transform(fl); nofl = selector.transform(nofl)
    fl = fl[:, np.argsort(scores)[::-1][0:N_features]]
    nofl = nofl[:, np.argsort(scores)[::-1][0:N_features]]
    minim = minim[np.argsort(scores)[::-1][0:N_features]]
    for i in range(fl.shape[1]):
      if minim[i] < 0:
        fl[:,i] = fl[:,i] + minim[i]; nofl[:,i] = nofl[:,i] + minim[i]
  elif method == 'pca':
    selector = PCA(n_components=N_features)
    selector.fit(np.concatenate((fl,nofl), axis=0))
    fl = selector.transform(fl); nofl = selector.transform(nofl)
    scores = selector.explained_variance_ratio_
    print "PCA was applied and ", np.shape(fl)[1], " components were kept."
    print "Variance explained: ", np.sum(selector.explained_variance_ratio_)
  #for i in range(N_features):  
    #print zip(np.arange(scores.shape[0]), (scores.argsort())[::-1], sorted(scores)[::-1])[i]      
  return fl, nofl, scores



def group_feature_selection(scores, depth=2, limit=500, Nattr=25, hist_is_in='yes'):
  """
  Selects features according to the group method.
  For every SHARP feature (e.g. TOTUSJH), select the statistical feature (e.g. max
  or mean) with the highest F-score, as long as this statistical feature has F score
  higher than a specific threshold. Repeat this procedure depth (e.g. 2 or 3) times.
  
  Args:
  scores: 1D array with size N_features which has the scores of the features
  depth: int, the number of times that we loop all the SHARP features to choose the best
  limit: float, the selection threshold for the F score of the Sharp features  
  Nattr: int, the number of SHARP features  
  hist_is_in: string, indicates where flaring history is one of the features
              If 'yes', it should be the last one
  
  Returns:
  selected_features: list of ints, the selected features
  """    
  NnewAttr = len(scores)
  if hist_is_in == 'yes':
    selected_attr = [int(NnewAttr-1)]
    scores = scores[0:-1]
    NnewAttr = len(scores)
  elif hist_is_in == 'no':
    selected_attr = []
  NstatAttr = NnewAttr/Nattr  
  rank_matrix=np.zeros((Nattr,NstatAttr))
  for i in range(Nattr):
    rank_matrix[i,:] = np.argsort(scores[[x for x in range(i,NnewAttr,Nattr)]])[::-1]
  
  for j in range(depth):
    for i in range(Nattr):
      if scores[Nattr*rank_matrix[i,j]+i] >= limit:
        selected_attr.append(int(Nattr*rank_matrix[i,j]+i))
  return selected_attr     

def normalize_stat_attr(fl, nofl, mode='mean'):
  """
  Nomalizes the statistical attributes (subtract mean, divide by stdev)
  
  Args:
  fl: 2D numpy array (Nobs, Nnewattr) of flaring-region features to be standardized
  nofl: 2D numpy array (Nobs, Nnewattr) of nonflaring-region features to be standardized
  mode: string, use of either "median" or "mean" for standardization

  Returns:
  2D numpy array (Nobs, Nnewattr) that is standardized
  """
  Nnewattr = np.shape(fl)[1]
  data=np.concatenate((fl,nofl), axis=0)
  for i in range(Nnewattr):
    if (mode == 'median'):
      ave = np.median(data[:,i]); sdev = np.std(data[:,i])
    elif (mode == 'mean'):   
      ave = np.mean(data[:,i]); sdev = np.std(data[:,i])
    elif (mode == 'middle'):
      ave = (np.min(data[:,i]) + np.max(data[:,i]))/2. 
      sdev = np.sqrt(np.mean(abs(data[:,i] - ave)**2))
    fl[:,i] -= ave; fl[:,i] /= sdev
    nofl[:,i] -= ave; nofl[:,i] /= sdev
  return (fl, nofl)
 


def log_rank_normalize_stat_attr(fl, nofl, mode='mean'):
  """
  Nomalizes the rank of statistical attributes with logistic function
  such that all features are between [-1,1] and have exactly the same distribution
  
  Args:
  fl: 2D numpy array (Nobs, Nnewattr) of flaring-region features to be standardized
  nofl: 2D numpy array (Nobs, Nnewattr) of nonflaring-region features to be standardized
  mode: string, use of either "median" or "mean" for standardization

  Returns:
  2D numpy array (Nobs, Nnewattr) that is standardized with logistic function
  """
  Nfl, Nnewattr = np.shape(fl); Nnofl = np.shape(nofl)[0]; Nobs = Nfl + Nnofl
  data=np.concatenate((fl,nofl), axis=0)
  for i in range(Nnewattr):
#    if (mode == 'median'):
#      ave = np.median(data[:,i])
#    elif (mode == 'mean'):   
#      ave = np.mean(data[:,i])
#    ave = (np.min(data[:,i]) + np.max(data[:,i]))/2. 
#    ave = np.median(data)
    order = data[:,i].argsort(); rank = order.argsort()
    for j in range(Nfl):  
#      fl[j,i] = (1. - np.exp(fl[j,i]-ave))/(1. + np.exp(fl[j,i]-ave))
      fl[j,i] = (1. - np.exp( 10*(rank[j]-Nobs/2.)/Nobs ))/(1. + np.exp( 10*(rank[j]-Nobs/2.)/Nobs ))
    for j in range(Nnofl):
      nofl[j,i] = (1. - np.exp( 10*(rank[j+Nfl]-Nobs/2.)/Nobs ))/(1. + np.exp( 10*(rank[j+Nfl]-Nobs/2.)/Nobs ))
  return (fl, nofl)



def log_normalize_stat_attr(fl, nofl, mode='middle'):
  """
  Nomalizes the statistical attributes with logistic function
  such that all features are between [-1,1]
  
  Args:
  fl: 2D numpy array (Nobs, Nnewattr) of flaring-region features to be standardized
  nofl: 2D numpy array (Nobs, Nnewattr) of nonflaring-region features to be standardized
  mode: string, can be: "median", "mean", "middle" for standardization

  Returns:
  2D numpy array (Nobs, Nnewattr) that is standardized with logistic function
  """
  Nfl, Nnewattr = np.shape(fl); Nnofl = np.shape(nofl)[0]
  #data=np.concatenate((fl,nofl), axis=0)
  #for i in range(Nnewattr):  
  #  fl[:,i] = fl[:,i] + np.min(data[:,i])
  #  nofl[:,i] = nofl[:,i] + np.min(data[:,i])   
  #  fl[:,i] = np.where(fl[:,i] > 0, np.log(fl[:,i]), 0)
  #  nofl[:,i] = np.where(nofl[:,i] > 0, np.log(nofl[:,i]), 0)
  data=np.concatenate((fl,nofl), axis=0)
  for i in range(Nnewattr):
    if (mode == 'median'):
      ave = np.median(data[:,i])
    elif (mode == 'mean'):   
      ave = np.mean(data[:,i])
    elif (mode == 'middle'):  
      ave = (np.min(data[:,i]) + np.max(data[:,i]))/2. 
    norm = 3.*np.std(data[:,i])   
    norm=1
    for j in range(Nfl):  
      fl[j,i] = (1. - np.exp((fl[j,i]-ave)/norm))/(1. + np.exp((fl[j,i]-ave)/norm))
     #fl[j,i] = (1. - np.exp( 10*(rank[j]-Nobs/2.)/Nobs ))/(1. + np.exp( 10*(rank[j]-Nobs/2.)/Nobs ))  
    for j in range(Nnofl):  
      nofl[j,i] = (1. - np.exp((nofl[j,i]-ave)/norm))/(1. + np.exp((nofl[j,i]-ave)/norm))
      #nofl[j,i] = (1. - np.exp( 10*(rank[j+Nfl]-Nobs/2.)/Nobs ))/(1. + np.exp( 10*(rank[j+Nfl]-Nobs/2.)/Nobs ))
  return (fl, nofl)

def oversample(data, ydata, n):
  """
  given a dataset (Nfl+Nnofl, Nattr) with Nfl flaring observ. and Nnofl nonfl. observ.,  
  creates dataset (n*Nfl+Nnofl, Nattr) which has n copies of the flaring obs. 
  plus all the original nonflaring observations
  
  Args:
  data: 2D np array (Nobs, Nattr)
  ydata: 1D np array (Nobs), 0 for noflare, 1 for flare
  n: int, the number of times that observations will be copied
  
  Returns:
  oversampled_data, 2D np.array (n*Nobs, Nattr)
  """
  fldata = data[np.where(ydata>0)[0],:]
  nofldata = data[np.where(ydata<1)[0],:]
  yfl = ydata[np.where(ydata>0)[0]]
  ynofl = ydata[np.where(ydata<1)[0]]  
  flshape = list(fldata.shape)
  flshape[0] = flshape[0]*n
  yflshape = list(yfl.shape)
  yflshape[0] = yflshape[0]*n
  newdata = np.concatenate((np.resize(fldata, tuple(flshape)), nofldata), axis=0)
  newydata = np.concatenate((np.resize(yfl, tuple(yflshape)), ynofl), axis=0)
  return (newdata, newydata)
 
 
def split_dataset(fl, nofl, test_size=0.3, mode='random'):
  """
  splits two datasets of flaring and non-flaring regions with given ratio of non-flaring (N)
  to flaring regions (P) into a training and a test set with same ratio of N/P
  
  Args:
  fl: 2D np array (Nobs, Nattr) of flaring regions
  nofl: 2D np array (Nobs, Nattr) of non-flaring regions
  test_size: float, the ratio of number of test observations to total number of observations
  mode: string, either 'random' or 'contiguous', indicates whether the fl, nofl datasets
    will be split randomly to form training and test sets or by preserving the order of the
    observations (for example first 7 in training set and next 3 in test set). Contiguous
    mode should be used for data that their ordering in non-arbitrary (e.g. nth observation
    may be correlated with n+1 observation). See also '3.1.3. A note on shuffling' at:
    http://scikit-learn.org/stable/modules/cross_validation.html
    
  Returns:
  train, test: 2D np arrays to be used for training and testing
  ytrain, ytest: 1D np arrays that contain the corresponding labels  
  """
  yfl = np.ones(np.shape(fl)[0]); ynofl = np.zeros(np.shape(nofl)[0])
  Nfl = np.shape(fl)[0]; Nnofl = np.shape(nofl)[0]  
  if mode == 'random':  
    fltrain, fltest, ytrain, ytest = train_test_split(fl, yfl, test_size=test_size)
    nofltrain, nofltest, ynotrain, ynotest = train_test_split(nofl, ynofl, test_size=test_size)
    train = np.concatenate((fltrain, nofltrain), axis=0)
    test = np.concatenate((fltest, nofltest), axis=0)
    ytrain = np.concatenate((ytrain, ynotrain), axis=0)
    ytest = np.concatenate((ytest, ynotest), axis=0)
    return train, test, ytrain, ytest
  elif mode == 'contiguous':
    Nfltrain = int(round(Nfl*(1.0-test_size))); Nfltest = Nfl - Nfltrain
    Nnofltrain = int(round(Nnofl*(1.0-test_size))); Nnofltest = Nnofl - Nnofltrain    
    
    fl = np.concatenate((fl, fl[0:Nfltrain-1]), axis=0)
    nofl = np.concatenate((nofl, nofl[0:Nnofltrain-1]), axis=0)
    
    index = int(round(Nfl*random.rand()))
    fltrain = fl[index:index+Nfltrain, :]; ytrain = np.ones(Nfltrain)
#    temp = index+Nnofltrain-Nnofl; start_index = max(0, temp)
    fltest = np.concatenate((fl[0:index, :], fl[index+Nfltrain:Nfl,:]), axis=0) 
    ytest = np.ones(Nfltest)

    index = int(round(Nnofl*random.rand()))
    nofltrain = nofl[index:index+Nnofltrain, :]; ynotrain = np.zeros(Nnofltrain)
#    temp = index+Nnofltrain-Nnofl; start_index = max(0, temp)
    nofltest = np.concatenate((nofl[0:index, :], nofl[index+Nnofltrain:Nnofl,:]), axis=0)
    ynotest = np.zeros(Nnofltest)
    
    train = np.concatenate((fltrain, nofltrain), axis=0)
    test = np.concatenate((fltest, nofltest), axis=0)
    ytrain = np.concatenate((ytrain, ynotrain), axis=0)
    ytest = np.concatenate((ytest, ynotest), axis=0)
    return train, test, ytrain, ytest


def confusion_table(pred, labels):
  """
  computes the number of TP, TN, FP, FN events given the arrays with predictions and true labels
  
  Args:
  pred: np array with predictions (1 for flare, 0 for nonflare)
  labels: np array with true labels (1 for flare, 0 for nonflare)
  
  Returns:
  Numbers of TP, TN, FP, FN events (floats)
  """  
  Nobs = len(pred)
  TN = 0.; TP = 0.; FP = 0.; FN = 0.
  for i in range(Nobs):
    if (pred[i] == 0 and labels[i] == 0):
      TN += 1
    elif (pred[i] == 1 and labels[i] == 0):
      FP += 1
    elif (pred[i] == 1 and labels[i] == 1):
      TP += 1 
    elif (pred[i] == 0 and labels[i] == 1):
      FN += 1
    else:
      print "Error! Observation could not be classified."
  return TP, TN, FP, FN    



def TSS_score(y_true, y_pred):
  TP, TN, FP, FN = confusion_table(y_pred, y_true)  
  return TP/(TP+FN) - FP/(FP+TN)    



def compute_scores(TP=0, TN=0, FP=0, FN=0, metrics=[]):
  """
  computes the values of the metrics that are passed to it
  
  Args:
  TP: float, Number of True Positive events
  TN: float, Number of True Negative events
  FP: float, Number of False Positive events
  FN: float, Number of False Negative events
  metrics: list of strings like 'TSS', 'acc', 'HSS2', the metrics to be computed

  Returns:
  np array with the values of metrics in the same order that they are passed  
  """
  if TP+TN+FP+FN > 0.0: 
    scores = np.zeros(len(metrics))
    if 'TSS' in metrics:
      index = metrics.index('TSS')  
      scores[index] = TP/(TP+FN) - FP/(FP+TN)    
    if 'HSS2' in metrics:
      index = metrics.index('HSS2')  
      scores[index] = 2.0*( (TP*TN) - (FN*FP) ) / ( (TP+FN)*(FN+TN) + (TN+FP)*(TP+FP) )
    if 'acc' in metrics:
      index = metrics.index('acc')  
      scores[index] = (TP + TN) / (TP + TN + FP + FN)
    return scores
  else:
    print 'Error! Metrics cannot be defined with zero events.'  
      

def print_results(t=0.0, TSS=np.array([]), HSS2=np.array([]), acc=np.array([])):
  """
  prints the mean value and the st dev of the predefined skill scores
  
  Args:
  t float, the total running time
  TSS: np.array, float, or int, True Skill Statistic
  HSS2: np.array, float, or int, Heidke Skill Score
  acc: np.array, float, or int, accuracyUsage

  
  Returns:
  Nothing
  """  
  if t > 0:
    print "Total running time = ", round(t, 3), " seconds."
  print
  if type(TSS) == np.ndarray and TSS.shape[0] > 0:
    print "Mean TSS = ", round(np.mean(TSS),4), " with error of ", round(np.std(TSS),4)
  elif type(TSS) == float or type(TSS) == int or type(TSS) == np.float64:
    print "TSS = ", round(TSS,4)
  if type(HSS2) == np.ndarray and HSS2.shape[0] > 0:  
    print "Mean HSS2 = ", round(np.mean(HSS2),4), " with error of ", round(np.std(HSS2),4)
  elif type(HSS2) == float or type(HSS2) == int or type(HSS2) == np.float64:
    print "HSS2 = ", round(HSS2,4)  
  if type(acc) == np.ndarray and acc.shape[0] > 0:  
    print "Mean Acc = ",  round(np.mean(acc),4), " with error of ", round(np.std(acc),4)
  elif type(acc) == float or type(acc) == int or type(acc) == np.float64:
    print "Acc = ", round(acc,4)
    
    
    
  
  