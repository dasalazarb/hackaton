# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:52:49 2021

@author: da.salazarb
"""


# compare standalone models for binary classification
import pandas as pd
# from numpy import mean
# from numpy import std
# from sklearn.datasets import make_classification
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
# from matplotlib import pyplot
from sklearn.ensemble import StackingClassifier
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('mlp', MLPClassifier(max_iter=1000, learning_rate='adaptive', random_state=10, early_stopping=True)))
    level0.append(('knn', KNeighborsClassifier(n_neighbors=5)))
    level0.append(('rf', RandomForestClassifier(random_state=1)))
    level0.append(('dt', DecisionTreeClassifier(max_depth=30,random_state=1)))
    level0.append(('svc', SVC(gamma=.01, kernel='rbf', probability=True,decision_function_shape='ovo')))
    level0.append(('bayes', GaussianNB()))
    # define meta learner model
    level1 = MLPClassifier(max_iter=1000, learning_rate='adaptive', random_state=10, early_stopping=True)
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
    
    # params_2grid = {'svc__C': [0.1, 1], 'svc__kernel': ["linear","poly","rbf","sigmoid"], 
    #                   'svc__degree': [2, 4], 'svc__gamma': [0.01,1]}
    params_2grid = {'classification__svc__C': [0.001,0.1, 10], 'classification__svc__kernel': ["linear","poly","rbf","sigmoid"], 
                     'classification__svc__degree': [3, 4, 6], 'classification__svc__gamma': [0.001,0.01,5], 
                     'classification__rf__n_estimators': [5, 15, 20], 'classification__rf__max_depth': [20,30], 
                     'classification__rf__max_features':["auto", 'sqrt', 'log2']}
    # , 'classification__mlp__hidden_layer_sizes':[(100,), (150, 100, 50), (75,50,25)]
    
    return model, params_2grid

def get_voting():
    clf1 = DecisionTreeClassifier(max_depth=30,random_state=0)
    forest = RandomForestClassifier(random_state=1)
    clf2 = KNeighborsClassifier(n_neighbors=5)
    clf3 = SVC(gamma=.01, kernel='rbf', probability=True,decision_function_shape='ovo')
    clf4 = GaussianNB()
    clf5 = MLPClassifier(max_iter=1000, learning_rate='adaptive', random_state=10, early_stopping=True)
    
    ## Voting
    model = VotingClassifier(estimators=[('svc', clf3), ('rf',forest), ('dt', clf1),
                                         ('gauss', clf4), ('knn', clf2), ('mlp', clf5)], 
                             voting='hard', weights=None, n_jobs=-1)
    
    # params_2grid = {'svc__C': [0.1, 1], 'svc__kernel': ["linear","poly","rbf","sigmoid"], 
    #                  'svc__degree': [2, 4], 'svc__gamma': [0.01,1]}
    params_2grid = {'classification__svc__C': [0.001,0.1, 10], 'classification__svc__kernel': ["linear","poly","rbf","sigmoid"], 
                     'classification__svc__degree': [3, 4, 6], 'classification__svc__gamma': [0.001,0.01,5], 
                     'classification__rf__n_estimators': [5, 15, 20], 'classification__rf__max_depth': [20,50], 
                     'classification__rf__max_features':["auto", 'sqrt', 'log2']}
    # , 'classification__mlp__hidden_layer_sizes':[(100,), (150, 100, 50), (75,50,25)]
    return model, params_2grid

# evaluate a given model using cross-validation
def evaluate_model(model, params_2grid, X, y, xtest, data_test):
    model = Pipeline([
        ('undersampling', SMOTE()),
        ('classification', model)
    ])
    # Training classifiers
    grid = GridSearchCV(estimator=model, param_grid=params_2grid, cv=5)
    
    # Fit model
    grid.fit(X, y)
    
    best_params = grid.best_params_
    # Predict
    pred1=grid.predict(xtest)
    pred1=pd.concat([data_test.CASENAME,pd.DataFrame(pred1)],axis=1)
    pred1.columns=["CASENAME","FLUIDTYPE"]
    
    return model, pred1, best_params