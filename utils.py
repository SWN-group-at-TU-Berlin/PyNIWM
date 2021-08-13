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

import time
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


import seaborn as sns 
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

import warnings; warnings.simplefilter('ignore')


##### Function to load dataset ###############################################

def load_data(filepath):
    
    dataset = pd.read_csv(filepath, header = 1)
    
    return dataset

##### Function for preparing the dataset for further computations ############

def data_prep(data, 
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
    
    # dict for choosing model 
    algorithms = {'RandomForestClassifier' : RandomForestClassifier,
                   'LGBMClassifier': LGBMClassifier,
                   }        # add ANN 
    
    # assigning number of folds
    skf = StratifiedKFold(n_splits= configfile['n-folds'])
    
    # initialize dict for best parameters
    best_params = {}
    
    # initialize matrix for all performances
    all_scores = pd.DataFrame()
    
    # set parameters for oversampling minority classes in dataset
    smoter = SMOTE(random_state=10, sampling_strategy='auto', k_neighbors = 6, n_jobs = -1)
    
    
    
    for i in configfile['algorithm']:
        
        # create parameter grid for algorithm
        grid = ParameterGrid(configfile['hyperParams'][i])
        
        # progress update for model
        print('Start cross-validation for ' + str(i))
        num_params = str(len(list(grid)))

        for count, params in  enumerate(grid):
            
            # progress update for parameter combinations
            print('Parameter combination '+str(count+1) +'/' + num_params )            

            # initialize list for prediction performance
            pred = []
            
            # progress update for folds
            progress_fold = tqdm(total = configfile['n-folds'], desc='Folds' )
            
            # initialize max time for model cross-validation
            start_time = time.time()
        
            if time.time() > start_time + configfile['max_time']:
                print(time.time())
                print('Max time has been reached.')
                break
        
            # split data for Cross-Validation
            for train_index, val_index in skf.split(X_cv, y_cv):
    
                X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
                y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]
                
                # Oversample only the data in the training section with SMOTE
                X_train_resample, y_train_resample =  smoter.fit_resample(X_train,y_train)
                
                
    
                # build model
                model = algorithms[i](**params)
                # fit model to train data
                model.fit(X_train_resample, y_train_resample)
                # predict on validation fold
                prediction = model.predict(X_val)
                Fscore = f1_score(y_val, prediction, average='micro')
                pred.append(Fscore)
                
                # progress update for folds
                progress_fold.update(1)
             
            # save results from the cv    
            all_scores = pd.concat([all_scores,pd.DataFrame([{'algorithm':i, 'params':params, 'cv_scores':pred, 'avg_score':np.mean(pred)}])])
               
        

     
            
        # merge all results together and choose best performing parameters
        if all_scores.empty: 
            print('The timelimit is too low, no results have been obtained.')
            break
        else:
            all_scores.reset_index(drop=True, inplace=True)
            max_score = all_scores.avg_score.loc[all_scores.algorithm == i].max()
            idxmax_score = all_scores.avg_score.loc[all_scores.algorithm == i].idxmax()
            
            best_params[i] = {'best_params': all_scores.params.iloc[idxmax_score], 
                               'F1 micro score': max_score}
        
    
    return all_scores, best_params


##### Function for running models with best parameters #######################

def run_best_model(algorithm, best_params, X_train, X_test, y_train, y_test):
    
    # dict for choosing model 
    algorithms = {'RandomForestClassifier' : RandomForestClassifier,
                   'LGBMClassifier': LGBMClassifier,
                   }        # add ANN 
    
    # initialize matrix for all performances
    best_scores = pd.DataFrame()
    
    # set parameters for oversampling minority classes in dataset
    smoter = SMOTE(random_state=10, sampling_strategy='not majority', k_neighbors = 6, n_jobs=-1)
    
    #Oversample only the data in the training section with SMOTE
    X_train_resample, y_train_resample =  smoter.fit_resample(X_train,y_train)
    
    for i in algorithm:
         # build model
         model = algorithms[i](**best_params[i]['best_params'])
         #fit model to train data
         model.fit(X_train_resample, y_train_resample)
         # predict on validation fold
         prediction = model.predict(X_test)
         
         # save prediction results for each algorithm
         best_scores = pd.concat([best_scores,pd.DataFrame([{'algorithm':i, 'prediction': prediction }])])
         
    best_scores.reset_index(drop=True, inplace=True)
    return best_scores
        

    
##### Function for displaying results ########################################

def plot_best_model(result_matrix, y_test):
    
    # Sorting fixturers in decreasing order of event counts
    fixtures,counts = np.unique(y_test, return_counts=True)
    count_sort_ind = np.argsort(-counts)
    
    # iterate over algorithms
    
    N = len(result_matrix)
    fig, axis = plt.subplots(nrows=N, ncols=1, sharey=True, figsize=(8*N,7*N), sharex = True)
    for i in range(N):
        
        # create  and plot confusion matrix
        cmNorm = confusion_matrix(y_test, result_matrix.iloc[i]['prediction'], normalize = 'true')
        heatmap = sns.heatmap(cmNorm[:, count_sort_ind][count_sort_ind], annot=True, fmt=".0%", cmap="Blues", ax=axis[i])
        heatmap.yaxis.set_ticklabels(fixtures[count_sort_ind], rotation=0, ha='right', fontsize=12)
        heatmap.xaxis.set_ticklabels(fixtures[count_sort_ind], rotation=45, ha='right', fontsize=12)
        axis[i].set_ylabel('True label', fontsize=14)
        axis[i].set_xlabel('Predicted label', fontsize=14)
        axis[i].set_title('Confusion Matrix for '+ result_matrix.iloc[i]['algorithm'], fontsize=16)
        b, t = axis[i].set_ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        axis[i].set_ylim(b, t) # update the ylim(bottom, top) values


    
    
    
    
    
    
    
    
    
    
    