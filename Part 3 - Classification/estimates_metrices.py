# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:11:43 2019

@author: lalit.h.suthar
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

dataset = load_digits()
X, y = dataset.data, dataset.target

for class_name, class_count in zip(dataset.target_names, np.bincount(dataset.target)):
    print(class_name,class_count)
    
# Creating a dataset with imbalanced binary classes:  
# Negative class (0) is 'not digit 1' 
# Positive class (1) is 'digit 1'
y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced != 1] = 0

print('Original labels:\t', y[1:30])
print('New binary labels:\t', y_binary_imbalanced[1:30])

np.bincount(y_binary_imbalanced) 

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

# Accuracy of Support Vector Machine classifier
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1).fit(X_train, y_train)
svm.score(X_test, y_test)


from sklearn.dummy import DummyClassifier

# Negative class (0) is most frequent
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
# Therefore the dummy 'most_frequent' classifier always predicts class 0
y_dummy_predictions = dummy_majority.predict(X_test)

y_dummy_predictions

dummy_majority.score(X_test, y_test)

svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm.score(X_test, y_test)
svm.predict(X_test)


from sklearn.metrics import confusion_matrix

# Negative class (0) is most frequent
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
y_majority_predicted = dummy_majority.predict(X_test)
confusion = confusion_matrix(y_test, y_majority_predicted)
tn, fp, fn, tp = confusion_matrix(y_test, y_majority_predicted).ravel()

print('Most frequent class (dummy classifier)\n', confusion)

total = tn + fp + fn + tp

# =============================================================================
# Accuracy: Overall, how often is the classifier correct?
# (TP+TN)/total = (100+407)/165 = 0.91
Accuracy = (tn + tp)/total

# Misclassification Rate: Overall, how often is it wrong?
# (FP+FN)/total = (10+5)/165 = 0.09
# equivalent to 1 minus Accuracy
# also known as "Error Rate"
Misclassification_Rate = Error_Rate = (fp + fn)/total

# True Positive Rate: When it's actually yes, how often does it predict yes?
# TP/actual yes = 100/105 = 0.95
# also known as "Sensitivity" or "Recall"
True_Positive_Rate = Sensitivity = Recall = tp / (tp + fn)

# False Positive Rate: When it's actually no, how often does it predict yes?
# FP/actual no = 10/60 = 0.17
False_Positive_Rate = fp / (tn + fp)

# True Negative Rate: When it's actually no, how often does it predict no?
# TN/actual no = 50/60 = 0.83
# equivalent to 1 minus False Positive Rate
# also known as "Specificity"
True_Negative_Rate = Specificity = tn / ( tn + fp)

# Precision: When it predicts yes, how often is it correct?
# TP/predicted yes = 100/110 = 0.91
Precision = tp / (fp + tp)

# Prevalence: How often does the yes condition actually occur in our sample?
# actual yes/total = 105/165 = 0.64
Prevalence = (fn + tp) / total
# =============================================================================



