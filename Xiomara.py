# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:28:03 2021

@author: da.salazarb
"""
# %% import libraries
from func_modelClass import *
import imblearn
import random
import sklearn
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# %% Load dataset
# data = pd.read_csv("C:/Users/da.salazarb/Downloads/ConcursoPetroleos/BASE/base_train_data.csv")
# data = pd.read_csv("C:/Users/ax.rodriguezc/Documents/HAckaton/202104-main/AVANZADO/adv_train_data.csv")
data = pd.read_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/adv_train_data.csv")
data_test = pd.read_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/adv_test_data.csv")
#data.shape

data.iloc[:,0:9].describe()

data.info()
data_test.info()

data.drop(data.columns[[31,32]],axis="columns",inplace=True)

data.describe(include=object)

data["FLUIDTYPE"].value_counts()
data["Aquifer"].value_counts()

pd.crosstab(data["FLUIDTYPE"], data["Aquifer"])

# for i in range(3):
#    sns.pairplot(data.iloc[:,random.sample(range(1, data.shape[1]), 5)])

# %% Data processing
# split into input and output elements
## quitar NANs
data.dropna(axis=0, subset=['FLUIDTYPE'], inplace=True); data.shape
## resetear los indices
data.reset_index(inplace=True, drop=True)
data1 = data.values
ix = [i for i in range(data1.shape[1]) if i != 11 and i != 13 and i != 0 and i != 42]
X, y = data1[:, ix], data1[:, 11]; X.shape
# define imputer
imputer = IterativeImputer(skip_complete=True)
# fit on the dataset
imputer.fit(X)
# transform the dataset
Xtrans = imputer.transform(X)
# print total missing
#print('Missing: %d' % sum(isnan(Xtrans).flatten()))

data2 = pd.concat([data.iloc[:,[0,42,13]], pd.DataFrame(Xtrans, columns=data.columns[ix])], axis=1)

data2.shape
data2.columns

boxplot =data2.boxplot(column=['Depth', 'PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo'])
#data3 = pd.concat([data2.iloc[:,0:3], np.log(data2.iloc[:,3:data2.shape[1]]+1)], axis=1)

# %% impute xtest
xtrain=data2[['PRESS','TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo']]
xtrain_scaled = StandardScaler().fit_transform(xtrain)

xtest=data_test[['PRESS', 'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo']]

imputer1 = IterativeImputer()
# fit on the dataset
imputer1.fit(xtest)
# transform the dataset
xtest_imputed = imputer1.transform(xtest)
xtest_imputed=pd.DataFrame(xtest_imputed)
xtest_imputed.columns=['PRESS','TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo']
xtest_imputed_scaled = StandardScaler().fit_transform(xtest_imputed)

#%% stacking and voting
model_stack, params_2grid_stack = get_stacking()
model_vot, params_2grid_vot = get_voting()

model_stack, pred_stack, best_params_stack = evaluate_model(model_stack, params_2grid_stack, xtrain_scaled, pd.Categorical(y), xtest_imputed_scaled, data_test)
model_vot, pred_vot, best_params_vot = evaluate_model(model_vot, params_2grid_vot, xtrain_scaled, pd.Categorical(y), xtest_imputed_scaled, data_test)

#%% xgboost
import xgboost; print(xgboost.__version__)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import optuna

def objective(trial):
    # Define the search space
    # criterions = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    gammas = trial.suggest_uniform('gamma', 0,0.4)
    subsamples = trial.suggest_uniform('subsample',0.6,0.9)
    colsample_bytrees = trial.suggest_uniform('colsample_bytree',0.6,0.9)
    reg_alphas = trial.suggest_int('reg_alpha',0,100)
    max_depths = trial.suggest_int('max_depth',3,9)
    min_child_weights = trial.suggest_int('min_child_weight',1,5)
    
    model = XGBClassifier(
     learning_rate =0.1,
     reg_alpha=reg_alphas,
     n_estimators=1000,
     max_depth=max_depths,
     min_child_weight=min_child_weights,
     gamma=gammas,
     subsample=subsamples,
     colsample_bytree=colsample_bytrees,
     objective = 'multi:softprob',
     eval_metric = 'mlogloss',
     nthread=4,
     use_label_encoder=False,
     # scale_pos_weight=1,
     seed=27)
    
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y)
    label_encoded_y = label_encoder.transform(y)
    model = Pipeline([
        ('undersampling', SMOTE()),
        ('xgboost_classifier', model)
    ])
    score = cross_val_score(model, xtrain, label_encoded_y, scoring="accuracy").mean()
    return score

study = optuna.create_study(study_name="xgboost_classifier",
                            direction="maximize",
                            sampler=TPESampler())

study.optimize(objective, n_trials=100)

# trial = study.best_trial
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y)
label_encoded_y = label_encoder.transform(y)
model = XGBClassifier(use_label_encoder=False, objective = 'multi:softprob', eval_metric = 'mlogloss',)
model.set_params(**study.best_params)
model.fit(xtrain, label_encoded_y)
# model = XGBClassifier()
# param_xgb = {}
# model.fit(xtrain, pd.Categorical(y))
predxgb=model.predict(xtest_imputed)
predxgb=pd.concat([data_test.CASENAME,pd.DataFrame(predxgb)],axis=1)
predxgb.columns=["CASENAME","FLUIDTYPE"]
predxgb.FLUIDTYPE = label_encoder.inverse_transform(predxgb.FLUIDTYPE)

#%%
from sklearn.cluster import KMeans

xtrain=data2[['Depth', 'Area', 'RESTHICK', 'PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo', 
       'POROSITY', 'NTG', 'PERM', 'CONWATER', 'PERM.AQUIFER', 'OOIP', 'RF',
       'RFmax', 'GOR', 'RELPERM.RESSAT.Kro',
       'RELPERM.ENDPOINT.Kro', 'RELPERM.ENDPOINT.Krw',
       'Field.OILRATE', 'Field.WATRATE', 'Field.GASRATE']]

xtest=data_test[['Depth', 'Area', 'RESTHICK', 'PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo',
       'POROSITY', 'NTG', 'PERM', 'CONWATER', 'PERM.AQUIFER', 'OOIP', 'RF',
       'RFmax', 'GOR', 'RELPERM.RESSAT.Kro',
       'RELPERM.ENDPOINT.Kro', 'RELPERM.ENDPOINT.Krw',
       'Field.OILRATE', 'Field.WATRATE', 'Field.GASRATE']]

imputer1 = IterativeImputer(max_iter=1000)
# fit on the dataset
imputer1.fit(xtrain)
# transform the dataset
xtest_imputed = imputer1.transform(xtest)
xtest_imputed=pd.DataFrame(xtest_imputed)
xtest_imputed.columns=['Depth', 'Area', 'RESTHICK', 'PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo',
       'POROSITY', 'NTG', 'PERM', 'CONWATER', 'PERM.AQUIFER', 'OOIP', 'RF',
       'RFmax', 'GOR', 'RELPERM.RESSAT.Kro',
       'RELPERM.ENDPOINT.Kro', 'RELPERM.ENDPOINT.Krw',
       'Field.OILRATE', 'Field.WATRATE', 'Field.GASRATE']

mean_RF=xtrain.RFmax.mean()
std_RF=xtrain.RFmax.std()

xtrain=(xtrain-xtrain.mean())/xtrain.std()
xtest_imputed=(xtest_imputed-xtest_imputed.mean())/xtest_imputed.std()

distortions = []
K = range(1,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(xtrain)
    distortions.append(kmeanModel.inertia_)
    
    
plt.figure(figsize=(8,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=0).fit(xtrain)
label_kmeans=kmeans.predict(xtest_imputed)


xtrain_labeled=pd.concat([pd.DataFrame(xtrain),pd.DataFrame(kmeans.labels_)], axis=1)
xtrain_labeled.columns = ['Depth', 'Area', 'RESTHICK', 'PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo',
       'POROSITY', 'NTG', 'PERM', 'CONWATER', 'PERM.AQUIFER', 'OOIP', 'RF',
       'RFmax', 'GOR', 'RELPERM.RESSAT.Kro',
       'RELPERM.ENDPOINT.Kro', 'RELPERM.ENDPOINT.Krw',
       'Field.OILRATE', 'Field.WATRATE', 'Field.GASRATE',
       'kmeans_label']

xtest_labeled=pd.concat([pd.DataFrame(xtest_imputed),pd.DataFrame(label_kmeans)], axis=1)
xtest_labeled.columns = ['Depth', 'Area', 'RESTHICK', 'PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo',
       'POROSITY', 'NTG', 'PERM', 'CONWATER', 'PERM.AQUIFER', 'OOIP', 'RF',
       'RFmax', 'GOR', 'RELPERM.RESSAT.Kro',
       'RELPERM.ENDPOINT.Kro', 'RELPERM.ENDPOINT.Krw',
       'Field.OILRATE', 'Field.WATRATE', 'Field.GASRATE',
       'kmeans_label']

ave_var=xtrain_labeled.groupby(by="kmeans_label").mean()
ave_var.RFmax*std_RF+mean_RF

#
xtest.loc[xtest_labeled.kmeans_label==0,"RF"]=ave_var.RFmax[0]*std_RF+mean_RF
xtest.loc[xtest_labeled.kmeans_label==1,"RF"]=ave_var.RFmax[1]*std_RF+mean_RF
xtest.loc[xtest_labeled.kmeans_label==2,"RF"]=ave_var.RFmax[2]*std_RF+mean_RF
xtest.loc[xtest_labeled.kmeans_label==3,"RF"]=ave_var.RFmax[3]*std_RF+mean_RF
xtest.loc[xtest_labeled.kmeans_label==4,"RF"]=ave_var.RFmax[4]*std_RF+mean_RF
#%%
pred1=pd.concat([pred_stack,xtest.RF],axis=1)
pred1.columns=["CASENAME","FLUIDTYPE","RF"]

pred2=pd.concat([pred_vot,xtest.RF],axis=1)
pred2.columns=["CASENAME","FLUIDTYPE","RF"]

pred1.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/pred1.csv", index=False)
pred2.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/pred2.csv", index=False)
# %%
predxgb=pd.concat([predxgb,xtest.RF],axis=1)
predxgb.columns=["CASENAME","FLUIDTYPE","RF"]
predxgb.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/predxgb.csv", index=False)
