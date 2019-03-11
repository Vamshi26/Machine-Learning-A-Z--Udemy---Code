# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 10:56:20 2019

@author: lalit.h.suthar
"""

'''
standardization of a given dataset is the process of removing its mean and scaling it to have unit of variance. 

Standardization is a common required operation for many machine learning estimators, as they require the data to closely resemble a standard normal distribution.

As such, conventional standardization involves centering and scaling the data in the given training set so that the set ends up with a mean of 0 and a standard deviation of 1.
'''

from sklearn.preprocessing import StandardScaler
'''
Standardize features by removing the mean and scaling to unit variance

The standard score of a sample x is calculated as:

z = (x - u) / s
where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False.

Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using the transform method.

Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

For instance many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the L1 and L2 regularizers of linear models) assume that all features are centered around 0 and have variance in the same order. 
If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.

This scaler can also be applied to sparse CSR or CSC matrices by passing with_mean=False to avoid breaking the sparsity structure of the data.
'''
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler().fit(data)
print("Fit data : ",data)
print("Scale : ",scaler.scale_)
print("Mean : ",scaler.mean_)
print("Variance : ",scaler.var_)
print("Transformed data : ",scaler.transform(data))
print(scaler.transform([[2, 2]]))


from sklearn.preprocessing import MinMaxScaler
'''
Transforms features by scaling each feature to a given range.

This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.

The transformation is given by:

X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
where min, max = feature_range.

This transformation is often used as an alternative to zero mean, unit variance scaling.
'''
data1 = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler1 = MinMaxScaler().fit(data1)
print("Fit data : ",data1)
print("Scale : ",scaler1.scale_)
print("Min : ",scaler1.min_)
print("data_max_ : ",scaler1.data_max_)
print("data_min_ : ",scaler1.data_min_)
print("feature_range : ",scaler1.feature_range)
print("Transformed data : ",scaler1.transform(data1))
print(scaler1.transform([[2, 2]]))



from sklearn.preprocessing import MaxAbsScaler
'''
Scale each feature by its maximum absolute value.

This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0. 
It does not shift/center the data, and thus does not destroy any sparsity.

This scaler can also be applied to sparse CSR or CSC matrices.
'''
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
transformer = MaxAbsScaler().fit(X)
print("Fit data : ",X)
print("Scale : ",transformer.scale_)
print("Max abs : ",transformer.max_abs_)
print("n_samples_seen_ : ",transformer.n_samples_seen_)
print("Transformed data : ",transformer.transform(X))