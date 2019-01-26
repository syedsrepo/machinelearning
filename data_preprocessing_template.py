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

dataset = pd.read_csv('Data.csv');

x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,3].values; 

"""
#missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0);
imputer.fit(x[:,1:3]); #upper bound to be excluded, use 3 instead of 2
x[:,1:3] = imputer.transform(x[:,1:3]);

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x = LabelEncoder();
x[:,0] = labelEncoder_x.fit_transform(x[:,0]);

oneHotEncoder = OneHotEncoder(categorical_features = [0])
x = oneHotEncoder.fit_transform(x).toarray()

labelEncoder_y = LabelEncoder();
y = labelEncoder_y.fit_transform(y);
"""

#training and testing sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

"""
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train);
x_test = sc_x.fit_transform(x_test);
"""
















