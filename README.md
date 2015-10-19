# ** Solar Flare Forecasting**

### **Summary**
This repository contains the Python code for making predictions of solar flares 24 hours in advance.

### **Data pre-processing**
Data preprocessing includes the following steps:
1. Read data for flaring and non-flaring regions (3D datacubes: (time, region, feature))
2. Clean data from NaNs
3. Gap filling (using a local average)
4. Standardization (zero median and standard deviation = 1)
5. Outlier removal (changes from one time frame to the next should be smaller than 10 standard deviations)
6. Standardization (after removing outliers)


### **Feature Extraction**
Hundreds of features are extracted from time series of vector magnetic field data. There are 25 features (magnetic free energy density, current helicity, Lorentz force etc), 4191 non-flaring regions and 254 flaring regions, and 61 hours of observations. For each feature and region, several statistical quantities are computed from the time series including: the mean, standard deviation, amplitude, maximum, minimum, skewness, kurtosis, slope of linear regression, and estimated value of the feature at the time of the flare using either linear or polynomial regression with Lasso.






