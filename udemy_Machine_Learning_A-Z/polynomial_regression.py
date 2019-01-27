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


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train);
x_test = sc_x.fit_transform(x_test);
"""

#simple linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x,y)

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


#visualising the simple linear regression results
plt.scatter(x,y, color='red');
plt.title("Polynomial Regression - Linear Regression Result")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.plot(x, lin_reg.predict(x))
plt.show()

#visualising the polynomial regression results

plt.scatter(x,y, color='red');
plt.title("Polynomial Regression - Polynomial Result")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.plot(x, lin_reg2.predict(x_poly))
plt.show()

#smooth curve
x_grid = np.arange(start=min(x),stop=max(x),step=0.1)
x_grid = x_grid.reshape(len(x_grid),1)
    
plt.scatter(x,y, color='red');
plt.title("Polynomial Regression - Polynomial Result, Internal 0.1")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)))
plt.show()

a = np.matrix([6.5]) 
lin_reg.predict(a)
lin_reg2.predict(poly_reg.fit_transform(a))
