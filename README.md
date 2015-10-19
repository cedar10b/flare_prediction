# **Solar Flare Forecasting**

### **Summary**
The goal of this project is to develop a predictive model for solar flares. The model is trained using almost 5 years of observations from the Solar Dynamics Observatory.

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


### **Feature Selection**
A univariate feature selection method based on F-score (1-way ANOVA) was employed. Other methods were also explored. The optimal number of features for best performance is around 70-80. This was determined with the validation method described below.


### **Performance Metric**
The dataset is unbalanced with a ratio of non-flaring to flaring regions of 16.5. For unbalanced datasets, accuracy is not a good metric since a classifier that always predicts "no flare" will have a very high accuracy but no practical application. In addition, the cost of type I and type II errors is not symmetric. Failing to predict a strong flare can be much more costly than having a false alarm. Therefore, an appropriate metric should give higher weight to recall than precision. The True Skill Statistic (TSS) defined as:

TSS = TP/P - FP/N

does not depend on the ratio of non-flaring to flaring observations and therefore it is suitable for comparing results from different studies. In addition, for unbalanced datasets (much higher N than P), TSS gives higher weight to Recall than precision since the fraction FP/N will be low even if FP is relatively high. For these reasons, TSS is considered the appropriate skill metric for this problem.


### **Machine Learning Algorithm**
A Support Vector Machine (SVM) with an RBF kernel is used for the classification. The choice of SVM is based on the following facts (a) all attributes are numerical, (b) the dataset is relatively clean without outliers,  (c) the dataset is unbalanced, (d) the decision boundary is probably highly nonlinear. Decision tree based methods (Random Forests, Gradient Boosting etc) are more suitable for datasets that have many categorical features or may have many outliers. Linear methods such as logistic regression have poor performance when the true decision boundary is highly nonlinear. In addition, SVM has a very natural way to counterbalance the high number of negative observations by adjusting the weights of each class. Here, the weight of positive class is set to 16.5 because the negative class has 16.5 times more observations. The parameters C and gamma were chosen with the validation method described below.


### **Validation**









