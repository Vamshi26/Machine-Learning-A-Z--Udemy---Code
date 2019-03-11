# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:54:02 2019
@author: lalit.h.suthar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
n = 15
X = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(X)+X/6 + np.random.randn(n)/10

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);
      
#part1_scatter()

def answer_one():
    X_pred = np.linspace(0,10,100)
    ''' 
    # linear regression
    lin_reg1 = LinearRegression()
    lin_reg1.fit(X_train[:,np.newaxis], y_train)
    y1_pred = lin_reg1.predict(X_pred[:,np.newaxis])
    
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_pred, y1_pred, label='test data')
    plt.legend(loc=4);
    '''
    
    pred_array = []
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    for i in [1,3,6,9]:
        poly_reg = PolynomialFeatures(degree = i)
        X_poly = poly_reg.fit_transform(X_train[:,np.newaxis])
        poly_reg.fit(X_poly, y_train)
        
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(X_poly, y_train)
        y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_pred[:,np.newaxis]))
        plt.plot(X_pred, y_pred, label='degree = {0}'.format(i))
        plt.legend(loc=1);
        plt.plot()
        pred_array.append(y_pred)
    
    return np.array(pred_array)

answer_one()

# feel free to use the function plot_one() to replicate the figure 
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    #%matplotlib notebook
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

#plot_one(answer_one())

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score
    global r2_train 
    r2_train = []
    global r2_test 
    r2_test = []
    for i in range(0,10):
        poly_reg = PolynomialFeatures(degree = i)
        X_poly = poly_reg.fit_transform(X_train[:,np.newaxis])
        X_poly_test = poly_reg.fit_transform(X_test[:,np.newaxis])
        #poly_reg.fit(X_poly, y_train)
        
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(X_poly, y_train)
        
        r2_train_temp = lin_reg_2.score(X_poly, y_train)
        r2_test_temp = lin_reg_2.score(X_poly_test, y_test)
        
        #y_train_pred = lin_reg_2.predict(poly_reg.fit_transform(X_train[:,np.newaxis]))
        #y_test_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test[:,np.newaxis]))
        
        r2_train.append( r2_train_temp)#r2_score(y_train_pred, y_train) )
        r2_test.append( r2_test_temp)#r2_score(y_test_pred, y_test) )
        
    r2_train = np.array(r2_train)
    r2_test = np.array(r2_test)
    return (r2_train, r2_test)

#o = answer_two()
#print(o)

def answer_three():
    #%matplotlib inline
    (r2train, r2test) = answer_two()
    plt.figure(figsize=(10,5))
    plt.plot(np.array(range(0,10)), r2train, label='train')
    plt.plot(np.array(range(0,10)), r2test, label='test')
    plt.xlabel("Degree")
    plt.ylabel("R^2 Score")
    plt.legend()
    plt.show()
    Underfitting = 0
    Overfitting = 9
    Good_Generalization = 6
    return (Underfitting, Overfitting, Good_Generalization) # 0,9,6
    
answer_three()
''' 
Question 4
Training models on high degree polynomial features can result in overly complex models that overfit, so we often use regularized versions of the model to constrain model complexity, as we saw with Ridge and Lasso linear regression.

For this question, train two models: a non-regularized LinearRegression model (default parameters) and a regularized Lasso Regression model (with parameters alpha=0.01, max_iter=10000) both on polynomial features of degree 12. Return the  R2R2  score for both the LinearRegression and Lasso model's test sets.

This function should return one tuple (LinearRegression_R2_test_score, Lasso_R2_test_score)
'''  
def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score
    global LinearRegression_R2_test_score
    global Lasso_R2_test_score
    
    poly_reg = PolynomialFeatures(degree = 12)
    X_poly = poly_reg.fit_transform(X_train[:,np.newaxis])
    poly_reg.fit(X_poly, y_train)
        
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y_train)
    LinearRegression_R2_test_score = lin_reg.score(poly_reg.fit_transform(X_test[:,np.newaxis]), y_test)
    
    lasso_reg = Lasso(alpha=0.01, max_iter=10000)
    lasso_reg.fit(X_poly, y_train)
    Lasso_R2_test_score = lasso_reg.score(poly_reg.fit_transform(X_test[:,np.newaxis]), y_test)

    return (LinearRegression_R2_test_score, Lasso_R2_test_score)

#answer_four()
