# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
# %% import libraries
import imblearn
import random
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

# %% Load dataset
# data = pd.read_csv("C:/Users/da.salazarb/Downloads/ConcursoPetroleos/BASE/base_train_data.csv")
data = pd.read_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/adv_train_data.csv")
data_test = pd.read_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/adv_test_data.csv")
#data.shape

data.iloc[:,0:9].describe()

data.info()

data.drop(data.columns[[31,32]],axis="columns",inplace=True)

data.describe(include=object)

data["FLUIDTYPE"].value_counts()
data["Aquifer"].value_counts()

pd.crosstab(data["FLUIDTYPE"], data["Aquifer"])

# for i in range(3):
#     sns.pairplot(data.iloc[:,random.sample(range(1, data.shape[1]), 5)])

# %% Data processing
# split into input and output elements
data.dropna(axis=0, subset=['FLUIDTYPE'], inplace=True); data.shape
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


#%% clasificacion base

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# example of random undersampling to balance the class distribution
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='majority')


# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=10,random_state=0)
forest = RandomForestClassifier(random_state=1)
clf2 = KNeighborsClassifier(n_neighbors=5)
clf3 = SVC(gamma=.01, kernel='rbf', probability=True,decision_function_shape='ovo')
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3),('rf',forest)],
                        voting='hard', weights=None)

xtrain=data2[['PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo']]

xtest=data_test[['PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo']]

imputer1 = IterativeImputer()
# fit on the dataset
imputer1.fit(xtest)
# transform the dataset
xtest_imputed = imputer1.transform(xtest)
xtest_imputed=pd.DataFrame(xtest_imputed)
xtest_imputed.columns=['PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo']

# %%
clf1.fit(xtrain, pd.Categorical(y))
clf2.fit(xtrain, pd.Categorical(y))
clf3.fit(xtrain, pd.Categorical(y))
eclf.fit(xtrain, pd.Categorical(y))
forest.fit(xtrain, pd.Categorical(y))


pred1=clf1.predict(xtest_imputed)
pred1=pd.concat([data_test.CASENAME,pd.DataFrame(pred1)],axis=1)
pred1.columns=["CASENAME","FLUIDTYPE"]

pred2=clf2.predict(xtest_imputed)
pred2=pd.concat([data_test.CASENAME,pd.DataFrame(pred2)],axis=1)
pred2.columns=["CASENAME","FLUIDTYPE"]

pred3=clf3.predict(xtest_imputed)
pred3=pd.concat([data_test.CASENAME,pd.DataFrame(pred3)],axis=1)
pred3.columns=["CASENAME","FLUIDTYPE"]


pred4=eclf.predict(xtest_imputed)
pred4=pd.concat([data_test.CASENAME,pd.DataFrame(pred4)],axis=1)
pred4.columns=["CASENAME","FLUIDTYPE"]

pred5=forest.predict(xtest_imputed)
pred5=pd.concat([data_test.CASENAME,pd.DataFrame(pred5)],axis=1)
pred5.columns=["CASENAME","FLUIDTYPE"]

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

mean_RF=xtrain.RF.mean()
std_RF=xtrain.RF.std()

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
       'Field.OILRATE', 'Field.WATRATE', 'Field.GASRATE','kmeans_label']



xtest_labeled=pd.concat([pd.DataFrame(xtest_imputed),pd.DataFrame(label_kmeans)], axis=1)
xtest_labeled.columns = ['Depth', 'Area', 'RESTHICK', 'PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo', 
       'POROSITY', 'NTG', 'PERM', 'CONWATER', 'PERM.AQUIFER', 'OOIP', 'RF',
       'RFmax', 'GOR', 'RELPERM.RESSAT.Kro',
       'RELPERM.ENDPOINT.Kro', 'RELPERM.ENDPOINT.Krw',
       'Field.OILRATE', 'Field.WATRATE', 'Field.GASRATE','kmeans_label']


ave_var=xtrain_labeled.groupby(by="kmeans_label").mean()
ave_var.RFmax*std_RF+mean_RF

#
xtest.loc[xtest_labeled.kmeans_label==0,"RF"]=ave_var.RFmax[0]*std_RF+mean_RF
xtest.loc[xtest_labeled.kmeans_label==1,"RF"]=ave_var.RFmax[1]*std_RF+mean_RF
xtest.loc[xtest_labeled.kmeans_label==2,"RF"]=ave_var.RFmax[2]*std_RF+mean_RF
xtest.loc[xtest_labeled.kmeans_label==3,"RF"]=ave_var.RFmax[3]*std_RF+mean_RF
xtest.loc[xtest_labeled.kmeans_label==4,"RF"]=ave_var.RFmax[4]*std_RF+mean_RF

#%%
pred1=pd.concat([pred1,xtest.RF],axis=1)
pred1.columns=["CASENAME","FLUIDTYPE","RF"]

pred2=pd.concat([pred2,xtest.RF],axis=1)
pred2.columns=["CASENAME","FLUIDTYPE","RF"]

pred3=pd.concat([pred3,xtest.RF],axis=1)
pred3.columns=["CASENAME","FLUIDTYPE","RF"]

pred4=pd.concat([pred4,xtest.RF],axis=1)
pred4.columns=["CASENAME","FLUIDTYPE","RF"]

pred5=pd.concat([pred5,xtest.RF],axis=1)
pred5.columns=["CASENAME","FLUIDTYPE","RF"]


pred1.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/pred1.csv", index=False)
pred2.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/pred2.csv", index=False)
pred3.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/pred3.csv", index=False)
pred4.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/pred4.csv", index=False)
pred5.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/pred5.csv", index=False)

#%%

data_prod = pd.read_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/adv_train_prod_data.csv")
#data_prod.loc[data_prod.CASENAME=="5a690972",:].BOPD

from scipy.integrate import simps
from numpy import trapz

coincidencias = list(set(list(data_prod.CASENAME.unique())).intersection(list(data2["CASENAME"])) )

## WOR
prod_total = pd.DataFrame(np.zeros([len(coincidencias),11]), columns=['CASENAME', 'oil', 'water', 'gas', 'wor', 'gor', 'oil30', 'water30', 'gas30', 'wor30', 'gor30'], index=coincidencias)
for i in coincidencias:
    prod_total.loc[i,"CASENAME"] = i
    prod_total.loc[i,"oil"] = trapz(data_prod.loc[data_prod.CASENAME==i,:].BOPD, dx=1,axis=0)
    prod_total.loc[i,"water"] = trapz(data_prod.loc[data_prod.CASENAME==i,:].BWPD, dx=1,axis=0)
    prod_total.loc[i,"gas"] = trapz(data_prod.loc[data_prod.CASENAME==i,:].MMSCFD, dx=1,axis=0)
    prod_total.loc[i,"wor"] = prod_total.loc[i,"water"]/(prod_total.loc[i,"water"]+prod_total.loc[i,"oil"])
    prod_total.loc[i,"gor"] = prod_total.loc[i,"gas"]/prod_total.loc[i,"oil"]

#%%
import numpy as np
import matplotlib.pyplot as plt

# Porcion de 30 dias
for i in coincidencias:
    data_temp_1=data_prod.loc[data_prod.CASENAME==i,"BOPD"]
    actual=0
    warning=0
    for index, item in reversed(list(enumerate(np.gradient(data_temp_1)))):
        if (item>0 and item < 0.5 and actual <0) or (item>0 and item < .5 and actual >0 and actual < .5):
            warning+=1
        if item>2 and actual <2:
            index_indentation=index 
        actual = item
    if warning>=15:
        prod_total.loc[i,"oil30"] = 0
        prod_total.loc[i,"water30"] = 0
        prod_total.loc[i,"gas30"] = 0
        prod_total.loc[i,"wor30"] = 0
        prod_total.loc[i,"gor30"] = 0
    else:
        prod_total.loc[i,"oil30"] = trapz(data_prod.loc[data_prod.CASENAME==i,:].BOPD[index_indentation:index_indentation+30], dx=1,axis=0)
        prod_total.loc[i,"water30"] = trapz(data_prod.loc[data_prod.CASENAME==i,:].BWPD[index_indentation:index_indentation+30], dx=1,axis=0)
        prod_total.loc[i,"gas30"] = trapz(data_prod.loc[data_prod.CASENAME==i,:].MMSCFD[index_indentation:index_indentation+30], dx=1,axis=0)
        if np.isnan(prod_total.loc[i,"water30"]/(prod_total.loc[i,"water30"]+prod_total.loc[i,"oil30"])):
            prod_total.loc[i,"wor30"] = 0
        else:
            prod_total.loc[i,"wor30"] = prod_total.loc[i,"water30"]/(prod_total.loc[i,"water30"]+prod_total.loc[i,"oil30"])
        if np.isnan(prod_total.loc[i,"gas30"]/prod_total.loc[i,"oil30"]):
            prod_total.loc[i,"gor30"] = 0
        else:
            prod_total.loc[i,"gor30"] = prod_total.loc[i,"gas30"]/prod_total.loc[i,"oil30"]

data_temp = pd.merge(data2,prod_total,on='CASENAME')

#%%
data_test = pd.read_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/adv_test_data.csv")
data_prod_test = pd.read_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/adv_test_prod_data.csv")

from scipy.integrate import simps
from numpy import trapz

sorted(list(data_prod_test.CASENAME.unique())) == sorted(list(data_test.CASENAME.unique()))

coincidencias = list(set(list(data_prod_test.CASENAME.unique())).intersection(list(data_test.CASENAME.unique())) )

## WOR
prod_total = pd.DataFrame(np.zeros([len(coincidencias),11]), columns=['CASENAME', 'oil', 'water', 'gas', 'wor', 'gor', 'oil30', 'water30', 'gas30', 'wor30', 'gor30'], index=coincidencias)
for i in coincidencias:
    prod_total.loc[i,"CASENAME"] = i
    prod_total.loc[i,"oil"] = trapz(data_prod_test.loc[data_prod_test.CASENAME==i,:].BOPD, dx=1,axis=0)
    prod_total.loc[i,"water"] = trapz(data_prod_test.loc[data_prod_test.CASENAME==i,:].BWPD, dx=1,axis=0)
    prod_total.loc[i,"gas"] = trapz(data_prod_test.loc[data_prod_test.CASENAME==i,:].MMSCFD, dx=1,axis=0)
    prod_total.loc[i,"wor"] = prod_total.loc[i,"water"]/(prod_total.loc[i,"water"]+prod_total.loc[i,"oil"])
    prod_total.loc[i,"gor"] = prod_total.loc[i,"gas"]/prod_total.loc[i,"oil"]

#%%
import numpy as np
import matplotlib.pyplot as plt

# Porcion de 30 dias
for i in coincidencias:
    data_temp_1=data_prod_test.loc[data_prod_test.CASENAME==i,"BOPD"]
    actual=0
    warning=0
    for index, item in reversed(list(enumerate(np.gradient(data_temp_1)))):
        if (item>0 and item < 0.5 and actual <0) or (item>0 and item < .5 and actual >0 and actual < .5):
            warning+=1
        if item>2 and actual <2:
            index_indentation=index 
        actual = item
    if warning>=15:
        prod_total.loc[i,"oil30"] = 0
        prod_total.loc[i,"water30"] = 0
        prod_total.loc[i,"gas30"] = 0
        prod_total.loc[i,"wor30"] = 0
        prod_total.loc[i,"gor30"] = 0
    else:
        prod_total.loc[i,"oil30"] = trapz(data_prod_test.loc[data_prod_test.CASENAME==i,:].BOPD[index_indentation:index_indentation+30], dx=1,axis=0)
        prod_total.loc[i,"water30"] = trapz(data_prod_test.loc[data_prod_test.CASENAME==i,:].BWPD[index_indentation:index_indentation+30], dx=1,axis=0)
        prod_total.loc[i,"gas30"] = trapz(data_prod_test.loc[data_prod_test.CASENAME==i,:].MMSCFD[index_indentation:index_indentation+30], dx=1,axis=0)
        if np.isnan(prod_total.loc[i,"water30"]/(prod_total.loc[i,"water30"]+prod_total.loc[i,"oil30"])):
            prod_total.loc[i,"wor30"] = 0
        else: 
            prod_total.loc[i,"wor30"] = prod_total.loc[i,"water30"]/(prod_total.loc[i,"water30"]+prod_total.loc[i,"oil30"])
        if np.isnan(prod_total.loc[i,"gas30"]/prod_total.loc[i,"oil30"]):
            prod_total.loc[i,"gor30"] = 0
        else:
            prod_total.loc[i,"gor30"] = prod_total.loc[i,"gas30"]/prod_total.loc[i,"oil30"]
            
# %%

data_temp = pd.merge(data_test,prod_total,on='CASENAME')
imputer1 = IterativeImputer(max_iter=1000)
# fit on the dataset
imputer1.fit(xtrain)
# transform the dataset

xtest = data_test
xtest_imputed = imputer1.transform(xtest)

xtest_imputed=pd.DataFrame(xtest_imputed)
# xtest_imputed.columns=['Depth', 'Area', 'RESTHICK', 'PRESS',
#        'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo',
#        'POROSITY', 'NTG', 'PERM', 'CONWATER', 'PERM.AQUIFER', 'OOIP', 'RF',
#        'RFmax', 'GOR', 'RELPERM.RESSAT.Kro',
#        'RELPERM.ENDPOINT.Kro', 'RELPERM.ENDPOINT.Krw',
#        'Field.OILRATE', 'Field.WATRATE', 'Field.GASRATE']

data_temp.dropna(subset=["RECOVERY"],axis=0,inplace=True)

ix = [i for i in range(data_temp.shape[1]) if i != 0 and i != 1 and i != 2]
xtrain_rec, y_rec = data_temp.iloc[:, ix], data_temp.iloc[:, 1]

# %%
# Training classifiers
clf1_2 = DecisionTreeClassifier(max_depth=10,random_state=0)
forest_2 = RandomForestClassifier(random_state=1)
clf2_2 = KNeighborsClassifier(n_neighbors=5)
clf3_2 = SVC(gamma=.01, kernel='rbf', probability=True,decision_function_shape='ovo')
eclf_2 = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3),('rf',forest)],
                        voting='hard', weights=None)

# %%
clf1_2.fit(xtrain_rec, pd.Categorical(y_rec))
clf2_2.fit(xtrain_rec, pd.Categorical(y_rec))
clf3_2.fit(xtrain_rec, pd.Categorical(y_rec))
eclf_2.fit(xtrain_rec, pd.Categorical(y_rec))
forest_2.fit(xtrain_rec, pd.Categorical(y_rec))


pred1_rec=clf1_2.predict(xtest_imputed)
pred1_rec=pd.concat([data_test.CASENAME,pd.DataFrame(pred1_rec)],axis=1)
pred1_rec.columns=["CASENAME","RECOVERY"]

pred2_rec=clf2_2.predict(xtest_imputed)
pred2_rec=pd.concat([data_test.CASENAME,pd.DataFrame(pred2_rec)],axis=1)
pred2_rec.columns=["CASENAME","RECOVERY"]

pred3_rec=clf3_2.predict(xtest_imputed)
pred3_rec=pd.concat([data_test.CASENAME,pd.DataFrame(pred3_rec)],axis=1)
pred3_rec.columns=["CASENAME","RECOVERY"]

pred4_rec=eclf_2.predict(xtest_imputed)
pred4_rec=pd.concat([data_test.CASENAME,pd.DataFrame(pred4_rec)],axis=1)
pred4_rec.columns=["CASENAME","RECOVERY"]

pred5_rec=forest_2.predict(xtest_imputed)
pred5_rec=pd.concat([data_test.CASENAME,pd.DataFrame(pred5_rec)],axis=1)
pred5_rec.columns=["CASENAME","RECOVERY"]