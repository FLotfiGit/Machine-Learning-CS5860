# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:18:54 2021

@author: eefat
"""
import math
import numpy as np
from sklearn.metrics import confusion_matrix as CM
import pandas as pd
import matplotlib.pyplot as plt

###### Entropy ##########################################
def get_entropy(y_predict, y_real):
    """
    Returns entropy of a split
    y_predict is the split decision, True/Fasle, and y_true can be multi class
    """

    n = len(y_real)
    ############################# true
    s_true=0
    dd = y_real[y_predict]
    ind = np.where(y_predict)
    #wt = w[y_predict]
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
    #wf = w[ind0]
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
    #s = np.sum(wt) * s_true + np.sum(wf) * s_false
    s = n_true*1.0/n * s_true + n_false*1.0/n * s_false # overall entropy, again weighted average
    return s
####### My Tree ###################################################
######### (class for tree) ########################################
class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth
    ########## (fit the tree) #################

    def fit(self, x, y, par_node={}, depth=0):
        if par_node is None: 
            return None
        elif len(y) == 0:
            return None
        elif self.all_same(y):
            return {'val':y[0]}
        elif depth >= self.max_depth:
            return None
        else: 
            col, cutoff, entropy = self.find_best_split_of_all(x, y)    # find one split given an information gain 
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
    ######### (find the split value in all features) #####################
    def find_best_split_of_all(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.find_best_split(c, y)
            if entropy == 0:    # find the first perfect cutoff. Stop Iterating
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy
    ######### (find the split value) ##################
    def find_best_split(self, col, y):
        min_entropy = 10
        n = len(y)
        for value in set(col):
            y_predict = col < value
            my_entropy = get_entropy(y_predict, y)
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
#############################################

        # test
############################################
######### data loading ##################
dfs = pd.read_csv('C:\MyFiles\UCCS_Courses\MachineLearning_SP2021\HW\HW2\Codes\Dataset\PimaIndiansDiabetesDatabase_kaggle\diabetes.csv') 

x=dfs.values[:,:-1]
y=dfs.values[:,-1]
m = len(y) 
print('Total Number of Training Example : ',m)
print('Number of Features : ',x.shape[1] )
print('Number of Labels : ', dfs.shape[1]-x.shape[1])

mytree = DecisionTreeClassifier(max_depth=1)
m = mytree.fit(x, y)
print('done!')


