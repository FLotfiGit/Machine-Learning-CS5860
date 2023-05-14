# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:32:47 2021

@author: eefat
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

##########################################
def MySVM(X1, Y1,lr,c,n_iters,nk):
    dfx = pd.DataFrame(X1)
    dfx['label']=Y1
    temp = dfx.sample(nk)
    X=temp.values[:,:-1]
    Y=temp.values[:,-1]
    ###############
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    m=len(Y)
    cost_history = np.zeros(n_iters)
    for i in range(n_iters):
        loss=0
        for idx, x_i in enumerate(X):
            condition = Y[idx] * (np.dot(x_i, w) - b) >= 1
            if condition:
                w -= (lr/m) * (1 * c * w)
            else:
                w -= (lr/m) * (1 * c * w - np.dot(x_i, Y[idx]))
                b -= (lr/m) * Y[idx]
            loss += max(0, (1-condition))
        cost_history[i]= (c*(np.linalg.norm(w)**2))+loss
    return w,b, cost_history

#---------------------------------------------------
    
def MySVMpredict(X,w,b):
    pred = np.dot(X, w) - b
    return np.sign(pred)    
####################################################
    ###         data            ###
####################################################
dfs = pd.read_excel('Skin_NonSkin.xlsx') 
#dfs = pd.read_excel('plrx.xlsx') 

ind=dfs[1]==1
dfs[1][ind]=-1
ind2=dfs[1]==2
dfs[1][ind2]=1
dfs[1] = dfs[1].astype(float, errors = 'raise')

########################################
x0=dfs.values[:,:-1]
y0=dfs.values[:,-1]

X_norm=x0
x_train,x_test,y_train,y_test=train_test_split(X_norm,y0,test_size=0.3)
x_test=np.array(x_test)
y_test=np.array(y_test) 

lr = 0.01
c = 0.001
n_iters = 1000
nk=400
w,b, cost_history = MySVM(x_train, y_train,lr,c,n_iters,nk)
print('thetha values= ',w)
print('b =', b)
y_pred = MySVMpredict(x_test,w,b)
 
acc = accuracy_score(y_test,y_pred)
print('MySVM Accuracy value = %', acc*100)


######################################
#### C effect on accuracy
"""
acc_v = np.zeros(8)
cc_v = [0.001,0.1,1,100,400,600,800,1000]
for i in range(8):
    c = cc_v[i]
    w,b, cost_history = MySVM(x_train, y_train,lr,c,n_iters,nk)
    y_pred = MySVMpredict(x_test,w,b)
    acc_v[i] = 2/np.linalg.norm(w)
    #accuracy_score(y_test,y_pred)
    #print('Accuracy value = %', acc*100)

plt.plot(cc_v,acc_v)
plt.grid()
plt.xlabel('C value')
plt.ylabel('\theta value')
plt.show()

"""

####################################
###   plot ###   ###   ###   ###
###################################
#"""
ind0 = np.where(y_test==1)
ind1 = np.where(y_test==-1)
xx00 = x_test[ind0,0]
xx01 = x_test[ind1,0]
xx10 = x_test[ind0,1]
xx11 = x_test[ind1,1]
xx20 = x_test[ind0,2]
xx21 = x_test[ind1,2]

"""
ind0 = np.where(y0==1)
ind1 = np.where(y0==-1)
xx00 = x0[ind0,0]
xx01 = x0[ind1,0]
xx10 = x0[ind0,1]
xx11 = x0[ind1,1]
"""

plt.scatter(xx00[0,:500],xx10[0,:500],marker='+')
plt.scatter(xx01[0,:500],xx11[0,:500],marker='_')
plt.grid()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

plt.plot(range(1,n_iters+1),cost_history)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Loss function')
plt.show()

######################################
######################################
########## Library ###################
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('--------------- result for library ---------------')
### Ad-aBoost, Nearest Neighbor classication, and Logistic Regression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

##### Ada-boost
"""
clf2 = AdaBoostClassifier(n_estimators=100,algorithm="SAMME")
clf2.fit(x_train,y_train)
y_pred2 = clf2.predict(x_test)

acc2 = accuracy_score(y_test,y_pred2)
print('AdaBoost Accuracy value_lib = %', acc2*100)
"""
##### KNN 
"""
model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_train,y_train)
y_pred2= model.predict(x_test) # 0:Overcast, 2:Mild

acc2 = accuracy_score(y_test,y_pred2)
print('KNN Accuracy value_lib = %', acc2*100)
"""
"""
##### logistic regression
logisticRegr = LogisticRegression(tol=0.001, C=0.01, max_iter=1000)
logisticRegr.fit(x_train, y_train)
y_pred2 = logisticRegr.predict(x_test)

acc2 = accuracy_score(y_test,y_pred2)
print('LR Accuracy value_lib = %', acc2*100)
"""

##### SVM 
model = SVC()
model.fit(x_train[0:3000,:], y_train[0:3000])
y_pred=model.predict(x_test)

acc2 = accuracy_score(y_test,y_pred)
print('Accuracy value_lib = %', acc2*100)
#################################

#plt.plot(range(1,n_iters+1),cost_history)
#plt.show()

"""

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(xx00[0,:500],xx10[0,:500],marker='+')
plt.scatter(xx01[0,:500],xx11[0,:500],marker='_')
x0_1 = [0,255]
x0_110 =-(w[0] * x0_1[0] + b + 0) / w[1]
x0_111 =-(w[0] * x0_1[1] + b + 0) / w[1]

#x0_12 =-(w[0] * x0_1 + b + 1) / w[1]
#x0_13 =-(w[0] * x0_1 + b - 1) / w[1]
plt.grid()

plt.plot(x0_1,[x0_110,x0_111], 'r--')
#plt.plot(x0_1,x0_12, 'k')
#plt.plot(x0_1,x0_13, 'k')
plt.show() 
#x1_min = np.amin(x_test[:,1])
#x1_max = np.amax(x_test[:,1])
#ax.set_ylim([x1_min-3,x1_max+3])
"""


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


###################################### plot
import seaborn as sn
data = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
plt.show()
#######################################################

