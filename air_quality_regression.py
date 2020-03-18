#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import BaggingRegressor

def Air_Quality_Regression(method):

    # Read in air quality data
    data = pd.read_csv('PRSA_data_2010.csv', sep=',')

    # Pick out month and PM2.5
    x0 = data.month
    y0 = data.pm25

    # Only pick out finite values
    x = x0[np.isfinite(y0) == True]
    y = y0[np.isfinite(y0) == True]

    X = np.zeros((len(x), 1))
    X[:,0] = x
    
    x_fit = np.zeros((101, 1))
    x_fit[:,0] = np.linspace(min(x0),max(x0),101)

    if (method == 'lin'):
        # Fit data using linear regression, which uses ordinary least squares
        reg = linear_model.LinearRegression()
        reg.fit(X, y)
        y_fit = reg.predict(x_fit)
    elif (method == 'poly'):
        # Fit data using polynomial regression, and using Ridge regression
        model = make_pipeline(PolynomialFeatures(3), Ridge())
        model.fit(X, y)
        y_fit = model.predict(x_fit)
    elif (method == 'SVR'):
        # Fit data using Support Vector Regression
        clf = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        clf.fit(X, y)
        y_fit = clf.predict(x_fit)
    elif (method == 'ensemble'):
        # Fit data using BaggingRegressor
        clf = BaggingRegressor(base_estimator=svm.SVR(),n_estimators=2, random_state=0).fit(X, y)
        y_fit = clf.predict(x_fit)
    else:
        print('Input method {0} not an available modelling option'.format(method))
        sys.exit(0)
        
    return x_fit, y_fit, x, y


if __name__ == '__main__':
    X_FIT, Y_FIT, X_DATA, Y_DATA = Air_Quality_Regression('poly')
    
    plt.plot(X_FIT, Y_FIT, color='k', label='fit')
    plt.scatter(X_DATA, Y_DATA, color='r', label='data')
    plt.minorticks_on()
    plt.xlabel('Month')
    plt.ylabel('PM 2.5')
    plt.legend()
    plt.show()
        
        

