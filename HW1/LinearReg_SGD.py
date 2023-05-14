# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 22:32:37 2021

@author: eefat
"""
## Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from numpy import random
from sklearn.model_selection import train_test_split

def compute_cost(X,y,theta):
    predict = X.dot(theta)
    error = np.subtract(predict,y)
    sq_error = np.squre(error)
    J = 1/(2*m)*np.sum(sq_error)
    return J


def GD(X,y,alpha,itterations):
    theta=np.zeros(X.shape[1])
    cost_history = np.zeros(itterations)
    m=len(y)
    for i in range(itterations):
        predict = X.dot(theta)
        error = np.subtract(predict,y)
        sum_delta = (alpha/m)*X.transpose().dot(error)
        theta = theta - sum_delta
        cost_history[i]=compute_cost(X,y,theta)
        
    return theta, cost_history
        






## SGD function ##########################################
def SGD(train_data,lrate,Maxiter,k,lam):
    
    w=np.zeros(shape=(1,train_data.shape[1]-1))
    b=0
    
    iter=1
    while(iter<=Maxiter): 

        temp=train_data.sample(k)
        y=np.array(temp['PE'])
        x=np.array(temp.drop('PE',axis=1))
        
        w_gradient=np.zeros(shape=(1,train_data.shape[1]-1))
        b_gradient=0
        
        for i in range(k): 
            prediction=np.dot(w,x[i])+b
            w_gradient=w_gradient+(-2)*x[i]*(y[i]-(prediction))+2*lam*(w)
            b_gradient=b_gradient+(-2)*(y[i]-(prediction))
        
        w=w-lrate*(w_gradient/k)
        b=b-lrate*(b_gradient/k)
        
        iter=iter+1
    return w,b 
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
#dfs = pd.read_csv('CASP.csv') 
#dfs1 =dfs[['RMSD']]
#dfs=dfs.append(dfs1)

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

x_train,x_test,y_train,y_test=train_test_split(X_norm,Y,test_size=0.3)

print('Number of Training Data : ',x_train.shape[0])
print('Number of Test Data : ',x_test.shape[0])

train_data = pd.DataFrame(x_train)
train_data['PE'] = y_train
train_data.head(3)

x_test=np.array(x_test)
y_test=np.array(y_test)    

## training   ##############3 if LSLR --> lam=0 , if Ridge Regression --> lam>0
w,b=SGD(train_data,lrate=0.01,Maxiter=1000,k=1,lam=0)


## testing 
y_hat_SGD=predict(x_test,w,b)
err = y_test - y_hat_SGD
## plot 
plt.scatter(y_test,y_hat_SGD)   # predict scatter 
plt.plot(y_test,y_test,Color='red')
plt.grid()
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Scatter plot from actual y and predicted y')
plt.show()
print('Mean Squared Error :',mean_squared_error(y_test, y_hat_SGD))


#plt.scatter(y_test,(err))
#plt.grid()
#plt.xlabel('Actual y')
#plt.ylabel('Predicted y')
#plt.title('Scatter plot from actual y and predicted y')
#plt.show()

#plt.plot(y_test,y_hat_SGD,color='red')
#plt.grid()
#plt.xlabel('Actual y')
#plt.ylabel('Predicted y')
#plt.title('Scatter plot from actual y and predicted y')
#plt.show()


   
