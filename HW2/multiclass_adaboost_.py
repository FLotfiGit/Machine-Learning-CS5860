# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 00:53:48 2021

@author: eefat
"""
import math
import numpy as np
from sklearn.metrics import confusion_matrix as CM
from sklearn import metrics 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

###### Entropy ##########################################
def get_entropy(y_predict, y_real,w):
    
    ############################# true
    s_true=0
    dd = y_real[y_predict]
    ind = np.where(y_predict)
    wt = w[y_predict]
    n_true = len(dd)
    classes = set(dd)
    for c in classes:   # for each class, get entropy
        n_c = sum(dd==c)
        c1 = sum(dd==c)
        c2=sum(dd!=c)
        if c1==0 or c2==0:
            e=0
        else:
            n1=c1+c2
            enp1 = -(c1*1.0/n1)*math.log(c1*1.0/n1, 2)
            enp2 = -(c2*1.0/n1)*math.log(c2*1.0/n1, 2)
            e = n_c*1.0/n_true * (enp1+enp2)     # weighted avg
        s_true += e
        
    ################################ false
    s_false=0
    dd0 = y_real[~y_predict]
    ind0=np.where(~y_predict)
    wf = w[ind0]
    n_false = len(dd0)
    classes0 = set(dd0)
    for c0 in classes0:   # for each class, get entropy
        n_c0 = sum(dd0==c0)
        c10 = sum(dd0==c0)
        c20=sum(dd0!=c0)
        if c10==0 or c20==0:
            e=0
        else:
            n10=c10+c20
            enp10 = -(c10*1.0/n10)*math.log(c10*1.0/n10, 2)
            enp20 = -(c20*1.0/n10)*math.log(c20*1.0/n10, 2)
            e = n_c0*1.0/n_false * (enp10+enp20)     # weighted avg
        s_false += e
    s = np.sum(wt) * s_true + np.sum(wf) * s_false
    #s = n_true*1.0/n * s_true + n_false*1.0/n * s_false # overall entropy, again weighted average
    return s
####### My Tree ###################################################
######### (class for tree) ########################################
class MyDecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth
    ########## (fit the tree) #################

    def fit(self, x, y,sample_weight, par_node={}, depth=0):
        if par_node is None: 
            return None
        elif len(y) == 0:
            return None
        elif self.all_same(y):
            return {'val':y[0]}
        elif depth >= self.max_depth:
            return None
        else: 
            col, cutoff, entropy = self.find_best_split_of_all(x, y,sample_weight)    # find one split given an information gain 
            y_left = y[x[:, col] < cutoff]
            y_right = y[x[:, col] >= cutoff]
            par_node = {'col': dfs.columns[col], 'index_col':col,
                        'cutoff':cutoff,
                       'val': np.round(np.mean(y))}
            par_node['left'] = self.fit(x[x[:, col] < cutoff], y_left, {}, depth+1)
            par_node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, {}, depth+1)
            self.depth += 1 
            self.trees = par_node
            return par_node
    ######### (find the split value in all features) #########
    def find_best_split_of_all(self, x, y,w):
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.find_best_split(c, y,w)
            if entropy == 0:    
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy
    ######### (find the split value) ##################
    def find_best_split(self, col, y,w):
        min_entropy = 10
        n = len(y)
        for value in set(col):
            y_predict = col < value
            my_entropy = get_entropy(y_predict, y,w)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff
    
    def all_same(self, items):
        return all(x == items[0] for x in items)
    ########### (prediction) ############################                                       
    def predict(self, x):
        tree = self.trees
        results = np.array([0]*len(x))
        for i, c in enumerate(x):
            results[i] = self._get_prediction(c)
        return results
    
    def _get_prediction(self, row):
        cur_layer = self.trees
        while cur_layer.get('cutoff'):
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']
            else:
                cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val')
##################################################################
           ###       ###      ###      ###       ###
##################################################################
class MySAMME:

    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.sample_weights = None

    def _check_X_y(self, X, y):
        assert set(y) == {-1, 1}, 'Response variable must be ±1'
        return X, y
    def fit(self, X, y, iters):
        
        n = X.shape[0]
        m=len(np.unique(y))
        self.sample_weights = np.zeros(shape=(iters, n))
        self.stumps = np.zeros(shape=iters, dtype=object)
        self.stump_weights = np.zeros(shape=iters)
        self.errors = np.zeros(shape=iters)
        
        # initialize weights uniformly
        self.sample_weights[0] = np.ones(shape=n) / n
        
        for t in range(iters):
            # fit  weak learner
            curr_sample_weights = self.sample_weights[t]
            
            stump = DecisionTreeClassifier(max_depth=1)
            stump = stump.fit(X, y, sample_weight=curr_sample_weights)
            
            # calculate error and stump weight from weak learner prediction
            stump_pred = stump.predict(X)
            err = curr_sample_weights[(stump_pred != y)].sum()# / n
            if err==0:
                err=1
            stump_weight = max(0,np.log((1 - err) / err)+np.log(m-1))
            
            # update sample weights
            new_sample_weights = (
                    curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
                    )
            new_sample_weights /= new_sample_weights.sum()
            
            # If not final iteration, update sample weights for t+1
            if t+1 < iters:
                self.sample_weights[t+1] = new_sample_weights
            # save results of iteration
            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            self.errors[t] = err
        return self
    
    def predict(self, X):
        """ Make predictions using already fitted model """
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        ##### new
        yy_r= np.zeros(shape=len(X))
        w=self.stump_weights
        #bb=np.argmax(val)
        for i in range(len(X)):
            v=np.unique(stump_preds[:,i])
            ws=np.zeros(shape=len(v))
            for i2 in range(len(v)):
               ws[i2]=w[stump_preds[:,i]==v[i2]].sum()
            yy_r[i] = v[np.argmax(ws)]
            #aa=stump_preds[bb,i] #yy[:,i]
            #aa=Counter(bb).most_common(1)
            #yy_r[i]=aa
        ##### new
        
        return yy_r #np.sign(np.dot(self.stump_weights, stump_preds))
#############################################################
        ##        ##         ##         ##          ##
######## data loading #######################################
#############################################################
        ##        ##         ##         ##          ##
######## data loading #######################################
dfs = pd.read_csv('Dataset\Multiclass\pageBlocks\dataset_30_page-blocks.csv')
#dfs = pd.read_csv('Dataset\Multiclass\shuttle\shuttle.csv')

x=dfs.values[:,:-1]
y=dfs.values[:,-1]

X_norm=x
x_train,x_test,y_train,y_test=train_test_split(X_norm,y,test_size=0.3)
x_test=np.array(x_test)
y_test=np.array(y_test) 

print('Number of Training Data : ',x_train.shape[0])
print('Number of Test Data : ',x_test.shape[0])


clf=MySAMME().fit(x_train,y_train,iters=60)
y_pred = clf.predict(x_test)

print("Performance:",100*sum(y_pred==y_test)/len(y_test))
print("Confusion Matrix:",CM(y_test,y_pred))

########## plot

import seaborn as sn
data = CM(y_test, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size

#######################################################
############## Sklearn ################################
#######################################################

clf2 = AdaBoostClassifier(n_estimators=20,algorithm="SAMME")
clf2.fit(x_train,y_train)
y_pred2 = clf2.predict(x_test)

print("Performance:",100*sum(y_pred2==y_test)/len(y_test))
print("Confusion Matrix:",CM(y_test,y_pred2)) 

###### plot
data2 = CM(y_test, y_pred2)
df_cm2 = pd.DataFrame(data2, columns=np.unique(y_test), index = np.unique(y_test))
df_cm2.index.name = 'Actual'
df_cm2.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm2, cmap="Blues", annot=True,annot_kws={"size": 16})# font size


# n_estimator =[2,5,10,20,30,40,50,60,70,100,200]
# performance = [84,86 ,92,92,92,99,99,96,99,99,99]

#per [89,90,90,90,...]
#per=[78,78]
