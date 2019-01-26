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

dataset = pd.read_csv('Position_Salaries.csv');

x = dataset.iloc[:,1:2].values;
y = dataset.iloc[:,2].values; 

#training and testing sets
"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
"""

#feature scaling
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x.reshape(-1, 1));
y = sc_y.fit_transform(y.reshape(-1, 1));
'''
#regression decision tree - Non-Linear and Non - continious
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion="mse", random_state=0)
regressor.fit(x,y)

#visualising the SVR results

x_grid = np.arange(start=min(x),stop=max(x),step=0.01)
x_grid = x_grid.reshape(len(x_grid),1)

plt.scatter(x,y, color='red');
plt.title("SVR")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.plot(x_grid, regressor.predict(x_grid))
plt.show()


a = np.matrix([6.5]) 
print(regressor.predict(a))
