# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 11:12:30 2021

@author: eefat
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 22:32:37 2021

@author: eefat
"""
## Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy import random
from sklearn.model_selection import train_test_split


def SGD(X1,y1,alpha,itterations,nk):
    dfx = pd.DataFrame(X1)
    dfx['label']=y1
    temp = dfx.sample(nk)
    X=temp.values[:,:-1]
    y=temp.values[:,-1]
    ###############
    theta=np.zeros(X.shape[1])
    b=0
    cost_history = np.zeros(itterations)
    m=len(y)
    for i in range(itterations):
        predict = X.dot(theta)+b
        error = np.subtract(predict,y)
        sum_delta = (alpha/m)*X.transpose().dot(error)
        theta = theta - sum_delta
        b = b - (alpha/m)*np.sum(error)
        cost_history[i]= np.sum(np.subtract(y,X.dot(theta)+b))/m  
        #print(cost_history[i])
    return theta,b, cost_history
        

def GD(X,y,alpha,itterations):
    theta=np.zeros(X.shape[1])
    b=0
    cost_history = np.zeros(itterations)
    m=len(y)
    for i in range(itterations):
        predict = X.dot(theta)+b
        error = np.subtract(predict,y)
        sum_delta = (alpha/m)*X.transpose().dot(error)
        theta = theta - sum_delta
        b = b - (alpha/m)*np.sum(error)
        cost_history[i]= np.sum(np.subtract(y,X.dot(theta)+b))/m  #compute_cost_LinearR(X,y,theta,b)
        #print(cost_history[i])
    return theta,b, cost_history
        
## Predict function #########################################
def predict(x,w,b):
    y_hat=[]
    for i in range(len(x)):
        y=np.asscalar(np.dot(w,x[i])+b)
        y_hat.append(y)
    return np.array(y_hat)
## Linear Regression with SGD ####################################
    #              #               #
######### data loading ############
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
## data split
x_train,x_test,y_train,y_test=train_test_split(X_norm,Y,test_size=0.3)

print('Number of Training Data : ',x_train.shape[0])
print('Number of Test Data : ',x_test.shape[0])

x_test=np.array(x_test)
y_test=np.array(y_test)    

## training   ############## 
itteration=1000
#theta,b,cost_hist = GD(x_train,y_train,0.35,itteration)
theta,b,cost_hist = SGD(x_train,y_train,0.01,itteration,nk=300)

## testing #################################################
y_hat=predict(x_test,theta,b)
## Evaluate metrics ########################################
MSEE = np.sum((y_test-y_hat)**2)/len(y_test)
MAE = np.sum(np.abs(y_test-y_hat))/len(y_test)
RMSE = np.sqrt(np.sum((y_test-y_hat)**2)/len(y_test))

y_bar = np.mean(y_test)
SSE = np.sum((y_test-y_hat)**2)
SST = np.sum((y_test-y_bar)**2)
SSR = np.sum((y_hat-y_bar)**2)
R_sq = 1-SSE/SST    
n = len(y_test)
AR_sq = 1-(SSE/n)/(SST/n)
MSR = SSR/8
MSE = SSE /(n-9)
FS = MSR/MSE
## plot  ####################################################
plt.plot(range(1, itteration +1), cost_hist)
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel("Number of iterations")
plt.ylabel("Error function (J)")
plt.title('Convergence of Stochastic Gradient Descent')
plt.show()


plt.scatter(y_test,y_hat)   
plt.plot(y_test,y_test,Color='red')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Scatter plot from actual y and predicted y')
plt.show()


print('Mean Squared Error :',MSEE)
print('Mean Absolute Error :',MAE)
print('Root Mean Square Error :',RMSE)
print('Coefficient of Determination R^2 :',R_sq)
print('Adjusted Coefficient of Determination R^2 :',AR_sq)
print('F statistics :',FS)

print('---------------------------------------------------------')
print('theta',theta)
print('bias',b)


   
