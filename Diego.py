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
#    sns.pairplot(data.iloc[:,random.sample(range(1, data.shape[1]), 5)])

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


#%% clasificacion base FLUIDTYPE

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




#%% clasificacion base FRMAX

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

#%% clasificacion avanzado RECOVERY (sin datos de produccion)

data3=data2.dropna(subset=["RECOVERY"],axis=0)

xtrain2=data3[['Depth', 'Area', 'RESTHICK', 'PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo', 
       'POROSITY', 'NTG', 'PERM', 'CONWATER', 'PERM.AQUIFER', 'OOIP', 
       'GOR', 'RELPERM.RESSAT.Kro',
       'RELPERM.ENDPOINT.Kro', 'RELPERM.ENDPOINT.Krw',
       'Field.OILRATE', 'Field.WATRATE', 'Field.GASRATE']]

xtest2=data_test[['Depth', 'Area', 'RESTHICK', 'PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo', 
       'POROSITY', 'NTG', 'PERM', 'CONWATER', 'PERM.AQUIFER', 'OOIP', 
       'GOR', 'RELPERM.RESSAT.Kro',
       'RELPERM.ENDPOINT.Kro', 'RELPERM.ENDPOINT.Krw',
       'Field.OILRATE', 'Field.WATRATE', 'Field.GASRATE']]

imputer1 = IterativeImputer()
# fit on the dataset
imputer1.fit(xtest2)
# transform the dataset
xtest_imputed2 = imputer1.transform(xtest2)
xtest_imputed2=pd.DataFrame(xtest_imputed2)
xtest_imputed2.columns=['Depth', 'Area', 'RESTHICK', 'PRESS',
       'TEMP', 'OILGRAV', 'SOLGOR', 'Visco', 'Psat', 'Bo', 
       'POROSITY', 'NTG', 'PERM', 'CONWATER', 'PERM.AQUIFER', 'OOIP', 
       'GOR', 'RELPERM.RESSAT.Kro',
       'RELPERM.ENDPOINT.Kro', 'RELPERM.ENDPOINT.Krw',
       'Field.OILRATE', 'Field.WATRATE', 'Field.GASRATE']


y2=data3.iloc[:, 1]

clf1.fit(xtrain2, pd.Categorical(y2))
clf2.fit(xtrain2, pd.Categorical(y2))
clf3.fit(xtrain2, pd.Categorical(y2))
eclf.fit(xtrain2, pd.Categorical(y2))
forest.fit(xtrain2, pd.Categorical(y2))



pred11=clf1.predict(xtest_imputed2)
pred11=pd.concat([data_test.CASENAME,pd.DataFrame(pred11)],axis=1)
pred11.columns=["CASENAME","RECOVERY"]

pred22=clf2.predict(xtest_imputed2)
pred22=pd.concat([data_test.CASENAME,pd.DataFrame(pred22)],axis=1)
pred22.columns=["CASENAME","RECOVERY"]

pred33=clf3.predict(xtest_imputed2)
pred33=pd.concat([data_test.CASENAME,pd.DataFrame(pred33)],axis=1)
pred33.columns=["CASENAME","RECOVERY"]


pred44=eclf.predict(xtest_imputed2)
pred44=pd.concat([data_test.CASENAME,pd.DataFrame(pred44)],axis=1)
pred44.columns=["CASENAME","RECOVERY"]

pred55=forest.predict(xtest_imputed2)
pred55=pd.concat([data_test.CASENAME,pd.DataFrame(pred55)],axis=1)
pred55.columns=["CASENAME","RECOVERY"]

#%%

pred1=pd.concat([pred1,pred11.RECOVERY],axis=1)
pred1.columns=["CASENAME","FLUIDTYPE","RF","RECOVERY"]

pred2=pd.concat([pred2,pred22.RECOVERY],axis=1)
pred2.columns=["CASENAME","FLUIDTYPE","RF","RECOVERY"]

pred3=pd.concat([pred3,pred33.RECOVERY],axis=1)
pred3.columns=["CASENAME","FLUIDTYPE","RF","RECOVERY"]

pred4=pd.concat([pred4,pred44.RECOVERY],axis=1)
pred4.columns=["CASENAME","FLUIDTYPE","RF","RECOVERY"]

pred5=pd.concat([pred5,pred55.RECOVERY],axis=1)
pred5.columns=["CASENAME","FLUIDTYPE","RF","RECOVERY"]

pred1.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/pred11.csv", index=False)
pred2.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/pred21.csv", index=False)
pred3.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/pred31.csv", index=False)
pred4.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/pred41.csv", index=False)
pred5.to_csv("C:/Users/da.salazarb/Desktop/202104-main/AVANZADO/pred51.csv", index=False)