# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:28:44 2021

@author: eefat
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

#################################################################
dfs = pd.read_excel('Folds5x2_pp.xlsx', sheetname=0,header=0,index_col=False,keep_default_na=True)
#dfs2 = pd.read_csv('CASP.csv') 
#dfs=dfs2[['F1','F2','F3','F4','F5','F6','F7','F8','F9','RMSD']]

X=dfs.values[:,:-1]
Y=dfs.values[:,-1]
m = len(Y) 
print('Total Number of Training Example : ',m)
print('Number of Features : ',X.shape[1] )
print('Number of Labels : ', dfs.shape[1]-X.shape[1])

## Features normalization
mu = np.mean(X, axis = 0)  
sigma = np.std(X, axis= 0, ddof = 1) 
X_norm = (X - mu)/sigma
#X_norm[:,0:1]=X[:,0:1]
## data split
x_train,x_test,y_train,y_test=train_test_split(X_norm,Y,test_size=0.3)

print('Number of Training Data : ',x_train.shape[0])
print('Number of Test Data : ',x_test.shape[0])

x_test=np.array(x_test)
y_test=np.array(y_test)    
####### Train ######################################################

### LSLR 
model_lslr =  linear_model.SGDRegressor(loss="squared_loss",eta0=.01,learning_rate="constant")
model_lslr.fit(x_train,y_train)
theta1 = model_lslr.coef_
b1 = model_lslr.intercept_
y_predict1 = model_lslr.predict(x_test)

plt.scatter(y_test,y_predict1,label="Linear Regression")
plt.plot(y_test,y_test,label="Best fit line",Color='red')
plt.legend(loc="upper left")
plt.grid()
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Scatter plot from actual y and predicted y')
plt.show()

MSEE1 = np.round(metrics.mean_squared_error(y_test,y_predict1),2)
print('LSLR Mean Squared Error',MSEE1)
print('---------------------------------------------------------')
print('LSLR theta',np.round(theta1,2))
print('LSLR bias',np.round(b1,2))



train_error = np.empty(6)
test_error = np.empty(6)
r=[0.0001,0.01,0.1,0.2,0.3,0.35] #np.linspace(0.005,0.05,100)

for i in range(len(r)):
    #model_r =  linear_model.ElasticNet(alpha=i, l1_ratio=i/10)
    model_r =  linear_model.SGDRegressor(loss="squared_loss",eta0=r[i],learning_rate="constant")
    model_r.fit(x_train,y_train)
    train_error[i] = metrics.mean_squared_error(y_train, model_r.predict(x_train))
    test_error[i] = mean_squared_error(y_test, model_r.predict(x_test))
plt.plot([0.0001,0.01,0.1,0.2,0.3,0.35], train_error, color='green', label='train')
plt.plot([0.0001,0.01,0.1,0.2,0.3,0.35], test_error, color='red', label='test')
#plt.ylim((0.0, 1e0))
plt.ylabel('mean squared error')
plt.xlabel('learning rate')
plt.legend(loc='upper left')