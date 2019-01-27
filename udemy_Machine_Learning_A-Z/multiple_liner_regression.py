#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 22:17:54 2019

@author: syedsalahuddin
"""

# data preprocessing

# importing the libraries

#mathematical calculations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir('/Applications/work/machinelearning')

#importing the dataset

dataset = pd.read_csv('50_Startups.csv');

x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,4].values; 

#encoding the categorial data , city values New York, california
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x = LabelEncoder();
x[:,3] = labelEncoder_x.fit_transform(x[:,3]);

oneHotEncoder = OneHotEncoder(categorical_features = [3])
x = oneHotEncoder.fit_transform(x).toarray()

#avoiding the dummyvariable trap
x = x[:,1:]

#prepare training and testing sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

"""
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train);
x_test = sc_x.fit_transform(x_test);
"""

#fit the linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression();
regressor.fit(x_train, y_train)

#perform the prediction
y_pred = regressor.predict(x_test)

import statsmodels.formula.api as sm
x = np.append(arr=np.ones((50,1)).astype(int), values=x, axis=1)

#backward Elimination
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog=x_opt).fit()
print(regressor_OLS.summary())

#with p-values and Adjusted R Squared:
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
#backward elemination 
'''
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
'''

def backwardElimination_alternate(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
'''
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
'''

















