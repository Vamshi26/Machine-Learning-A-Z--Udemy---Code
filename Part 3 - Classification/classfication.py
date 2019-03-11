# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 21:10:16 2019

@author: lalit.h.suthar
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

mush_df = pd.read_csv('mushrooms.csv',error_bad_lines=False)
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

def answer_five():
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train2, y_train2)
    
    features = []
    for f, i in zip(X_train2.columns, clf.feature_importances_):
        features.append((i, f))
    features.sort(reverse=True)
    
    return [feature[1] for feature in features[:5]]

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve
    
    svclf = SVC(kernel='rbf', C=1, random_state=0)
    gamma = np.logspace(-4,1,6)
    train_scr, test_scr = validation_curve(svclf, X_subset, y_subset, param_name='gamma', param_range=gamma, scoring='accuracy')
    
    train_scr_mean = train_scr.mean(axis=1)
    test_scr_mean = test_scr.mean(axis=1)
    return (train_scr_mean, test_scr_mean)



def answer_seven():
    import matplotlib.pyplot as plt
    train_scr_mean, test_scr_mean = answer_six()
    gamma = np.logspace(-4,1,6)
    plt.figure()
    plt.plot(gamma, train_scr_mean, 'b--', gamma, test_scr_mean, 'g-')
    plt.xlabel('Gamma')
    plt.ylabel('Accuracy Score')
    return (0.001, 10, 0.1) #(Underfitting, Overfitting, Good_Generalization)