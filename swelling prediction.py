# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:43:47 2023

@author: Artur
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import statistics

from sklearn.preprocessing import PolynomialFeatures
#df = pd.read_csv("data_to_use_only clays_PCA_without_f.csv")

df = pd.read_csv("filename.csv")
#df.drop('No', inplace = True,axis=1)
  
#print(df.head())

X = df.drop('Swelling',axis= 1)
y = df['Swelling']

######################
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


######################

sns.pairplot(df)

#####################

#Single shot#
poly = PolynomialFeatures(degree=1, include_bias=False)
poly_features = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.2, random_state=20)

poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)

poly_reg_y_predicted = poly_reg_model.predict(X_test)

poly_reg_rmse = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))

poly_reg_rmse

######################3
# state variation

RMSE = []
R2 = []
list_a = range(0,1000)
for i in list_a:
    poly = PolynomialFeatures(degree=1, include_bias=False)
    poly_features = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.2, random_state=i)

    poly_reg_model = LinearRegression()
    poly_reg_model.fit(X_train, y_train)

    poly_reg_y_predicted = poly_reg_model.predict(X_test)

    poly_reg_rmse = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))
    poly_reg_rmse
    RMSE.append(poly_reg_rmse)
    r2_score(y_test, poly_reg_y_predicted)
    R2.append(r2_score(y_test, poly_reg_y_predicted))
print(statistics.mean(RMSE))
print(statistics.mean(R2))
