#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 10:16:26 2021

@author: Marie-Philine
"""

#### Packages ################################################################
import numpy as np
import pandas as pd

from ruamel.yaml import YAML

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


import seaborn as sns 
import matplotlib as plt

import warnings; warnings.simplefilter('ignore')


##### Function to load dataset ###############################################

def load_data(filepath):
    
    dataset = pd.read_csv(filepath, header = 1)
    
    return dataset

##### Function for preparing the dataset for further computations ############

def dataPrep(data, 
             featList, 
             labelName, randomState, testSize):
    
    '''
    This function defines the label (target variable) and the feature list
    as well as splitting the dataset into test and training set.
    Default values are set to the presented paper.
    '''
    # Shuffle data
    np.random.seed(10)
    allData_shuffled = data.iloc[np.random.permutation(len(data))]
    allData_shuffled = allData_shuffled.reset_index(drop=True)
    # Dividing X and Y
    X = allData_shuffled[featList]
    X = pd.get_dummies(X, columns = ['Weekday'])
    y = allData_shuffled[labelName] 
    
    # Train - test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= testSize,
                                                        random_state= randomState)

    
    return  X_train, X_test, y_train, y_test

##### Function for reading the configuration file ############################

def read_config(filename):
    
    with open(filename) as f:
        yaml = YAML(typ='safe')
        cfg = yaml.load(f)
        
    return cfg


##### Function for running the CV on all selected models #####################

def run_CV(configfile, X_cv, y_cv):

    skf = StratifiedKFold(n_splits= configfile['n-folds'])
    
    # initialize dict for best parameters
    best_params = {}
    
    # initialize matrix for all performances
    all_scores = pd.DataFrame()
    
    for i in configfile['algorithm']:
        
       
        
        # create parameter grid for algorithm
        grid = ParameterGrid(configfile['hyperParams'][i])
        
        
        
        for params in grid:
            
            # initialize list for prediction performance
            pred = []
        
            # split data for Cross-Validation
            for train_index, val_index in skf.split(X_cv, y_cv):
    
                X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
                y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]
    
                # build model
                model = globals()[i](**params)
                # fit model to train data
                model.fit(X_train, y_train)
                # predict on validation fold
                prediction = model.predict(X_val)
                Fscore = f1_score(y_val, prediction, average='micro')
                pred.append(Fscore)
                
            all_scores = pd.concat([all_scores,pd.DataFrame([{'algorithm':i, 'params':params, 'cv_scores':pred, 'avg_score':np.mean(pred)}])])
            
        #    
        all_scores.reset_index(drop=True, inplace=True)
        max_score = all_scores.avg_score.loc[all_scores.algorithm == i].max()
        idxmax_score = all_scores.avg_score.loc[all_scores.algorithm == i].idxmax()
        
        best_params[i] = {'best_params': all_scores.params.iloc[idxmax_score], 
                           'F1 micro score': max_score}
    
    
    return all_scores, best_params


##### Function for running models with best parameters #####################

def run_bestModel(algorithm, best_params, X_train, X_test, y_train, y_test):
    
    # initialize matrix for all performances
    best_scores = pd.DataFrame()
    
    for i in algorithm:
         # build model
         model = globals()[i](**best_params[i]['best_params'])
         #fit model to train data
         model.fit(X_train, y_train)
         # predict on validation fold
         prediction = model.predict(X_test)
         Fscore = f1_score(y_test, prediction, average='micro')
         
         best_scores = pd.concat([best_scores,pd.DataFrame([{'algorithm':i, 'F1 score':Fscore }])])
         
    best_scores.reset_index(drop=True, inplace=True)
    ## add confusion matrix and feature importance
    return best_scores
        

    
    
    
    
    
    
    
    
    
    
    
    