#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 09:48:57 2019

@author: syedsalahuddin
"""

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

dataset = pd.read_csv('Salary_Data.csv');

x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,1].values; 

#training and testing sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# feature scaling is not required

#fit the model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression();
regressor.fit(x_train,y_train)

#predict the test set results
y_pred =  regressor.predict(x_test);

#visualize the train set results
plt.scatter(x_train,y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train))
plt.title('Years of Experiance VS Salary')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()

#visulize the test set results
plt.scatter(x_test,y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train))
plt.title('Years of Experiance VS Salary')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()

















