# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:14:36 2021

@author: eefat
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
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
X_norm=X
## data split
x_train,x_test,y_train,y_test=train_test_split(X_norm,Y,test_size=0.3)

print('Number of Training Data : ',x_train.shape[0])
print('Number of Test Data : ',x_test.shape[0])

x_test=np.array(x_test)
y_test=np.array(y_test)    
####### Train ######################################################

### LSLR 
model_lslr =  linear_model.LinearRegression(normalize=True)
model_lslr.fit(x_train,y_train)
theta1 = model_lslr.coef_
b1 = model_lslr.intercept_
y_predict1 = model_lslr.predict(x_test)

### Ridge R 
model_ridge =  linear_model.Ridge(normalize=True, alpha=0.1)
model_ridge.fit(x_train,y_train)
theta2 = model_ridge.coef_
b2 = model_ridge.intercept_
y_predict2 = model_ridge.predict(x_test)
print('max itteration Ridge', model_ridge.n_iter_)

### Lasso R 
model_lasso =  linear_model.Lasso(normalize=True, alpha=0.01)
model_lasso.fit(x_train,y_train)
theta3 = model_lasso.coef_
b3 = model_lasso.intercept_
y_predict3 = model_lasso.predict(x_test)
print('max itteration Lasso', model_lasso.n_iter_)

### Elastic Net R
model_Elastic =  linear_model.ElasticNet(alpha=0.1,l1_ratio=0.1)
model_Elastic.fit(x_train,y_train)
theta4 = model_Elastic.coef_
b4 = model_Elastic.intercept_
y_predict4 = model_Elastic.predict(x_test)
print('max itteration Elastic Net', model_Elastic.n_iter_)

##### Test ##############################33333

plt.scatter(y_test,y_predict1,label="Linear Regression")
plt.scatter(y_test,y_predict2,label="Ridge Regression")
plt.scatter(y_test,y_predict3,label="Lasso Regression")
plt.scatter(y_test,y_predict4,label="Elastic Net Regression")
plt.plot(y_test,y_test,label="Best fit line")
plt.legend(loc="upper left")
plt.grid()
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Scatter plot from actual y and predicted y')
plt.show()
##############################################################
MSEE1 = np.round(metrics.mean_squared_error(y_test,y_predict1),2) #np.sum((y_test-y_predict)**2)/len(y_test)
MSEE2 = np.round(metrics.mean_squared_error(y_test,y_predict2),2)
MSEE3 = np.round(metrics.mean_squared_error(y_test,y_predict3),2)
MSEE4 = np.round(metrics.mean_squared_error(y_test,y_predict4),2)

train_error = np.empty(40)
test_error = np.empty(40)
for i in range(40):
    
    #model_r =  linear_model.ElasticNet(alpha=i, l1_ratio=i/10)
    model_r =  linear_model.Ridge(normalize=True, alpha=i/10)
    model_r.fit(x_train,y_train)
    train_error[i] = metrics.mean_squared_error(y_train, model_r.predict(x_train))
    test_error[i] = mean_squared_error(y_test, model_r.predict(x_test))
plt.plot(np.arange(40)/10, train_error, color='green', label='train')
plt.plot(np.arange(40)/10, test_error, color='red', label='test')
#plt.ylim((0.0, 1e0))
plt.ylabel('mean squared error')
plt.xlabel('$\lambda$')
plt.legend(loc='upper left')

print('LSLR Mean Squared Error',MSEE1)
print('Ridge Mean Squared Error',MSEE2)
print('Lasso Mean Squared Error',MSEE3)
print('Elastic Net Mean Squared Error',MSEE4)

print('---------------------------------------------------------')
print('LSLR theta',np.round(theta1,2))
print('LSLR bias',np.round(b1,2))
print('---------------------------------------------------------')
print('Ridge theta',np.round(theta2,2))
print('Ridge bias',np.round(b2,2))
print('---------------------------------------------------------')
print('Lasso theta',np.round(theta3,2))
print('Lasso bias',np.round(b3,2))
print('---------------------------------------------------------')
print('Elastic Net theta',np.round(theta4,2))
print('Elastic Net',np.round(b4,2))