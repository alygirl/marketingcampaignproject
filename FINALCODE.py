# -*- coding: utf-8 -*-
"""
Created on Sun May 16 15:16:18 2021

@author: alyso
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 13 08:50:21 2021

@author: alyso
"""

#import packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#read in data
path="C:\\Users\\alyso\\Documents\\Aly\\Quickenloans Case study\\"
os.chdir(path)

data_orig=pd.read_csv(path+"DSA Data Set.csv")
data = data_orig.copy()
#check variables are reading in with the correct dtypes              
data.dtypes

#check for null fields
data.isnull().sum()

#%%%%%
#EXAMINE VARIABLES INDEPENDANTLY

#examine the target variable
sns.catplot(x='y', kind='count', data=data)
data['y'].value_counts()



####plot numerical variables

fig, ax = plt.subplots(5, 2,figsize=(20,20))

sns.boxplot( y="age",  data=data,ax=ax[0,0])
sns.boxplot( y="duration",  data=data,ax=ax[0,1])
sns.boxplot( y="campaign", data=data,ax=ax[1,0])
sns.boxplot( y="pdays",  data=data,ax=ax[1,1])
sns.boxplot( y="previous",  data=data,ax=ax[2,0])
sns.boxplot( y="emp.var.rate",  data=data,ax=ax[2,1])
sns.boxplot( y="cons.price.idx",  data=data,ax=ax[3,0])
sns.boxplot( y="cons.conf.idx", data=data,ax=ax[3,1])
sns.boxplot( y="euribor3m",  data=data,ax=ax[4,0])
sns.boxplot( y="nr.employed",  data=data,ax=ax[4,1])

plt.savefig('num outliers.png')

#get numeric variable stats
stats=data.describe()
stats.to_excel('stats.xlsx')
data['pdays'].value_counts()

#look into select numeric variables
data.groupby(['poutcome','pdays'])['poutcome'].count()
data['pdays'][(data['pdays'] != 999)].describe()
data['previous'][(data['previous'] != 0)].describe()


#####plot categorical variables for outliers
fig, ax = plt.subplots(5, 2,figsize=(30,30))
fig.tight_layout(h_pad=10)
plt.subplots_adjust( bottom=0.2)
data['job'].value_counts().plot(kind="bar",ax=ax[0,0])
data['marital'].value_counts().plot(kind="bar",ax=ax[0,1])
data['education'].value_counts().plot(kind="bar",ax=ax[1,0])
data['default'].value_counts().plot(kind="bar",ax=ax[1,1])
data['housing'].value_counts().plot(kind="bar",ax=ax[2,0])
data['loan'].value_counts().plot(kind="bar",ax=ax[2,1])
data['contact'].value_counts().plot(kind="bar",ax=ax[3,0])
data['month'].value_counts().plot(kind="bar",ax=ax[3,1])
data['day_of_week'].value_counts().plot(kind="bar",ax=ax[4,0])
data['poutcome'].value_counts().plot(kind="bar",ax=ax[4,1])

ax[0, 0].set_title('job')
ax[0, 1].set_title('marital')
ax[1, 0].set_title('education')
ax[1, 1].set_title('default')
ax[2, 0].set_title('housing')
ax[2, 1].set_title('loan')
ax[3, 0].set_title('contact')
ax[3, 1].set_title('month')
ax[4, 0].set_title('day_of_week')
ax[4, 1].set_title('poutcome')
plt.savefig('cat outliers.png')

data['job'].value_counts().plot(kind="bar")


#%%%%%
#EXAMINE VARIABLES VS TARGET

#numeric variables 
fig, ax = plt.subplots(5, 2,figsize=(20,20))

sns.boxplot(x="y", y="age",  data=data,ax=ax[0,0])
sns.boxplot(x="y", y="duration",  data=data,ax=ax[0,1])
sns.boxplot(x="y", y="campaign", data=data,ax=ax[1,0])
sns.boxplot(x="y", y="pdays",  data=data,ax=ax[1,1])
sns.boxplot(x="y", y="previous",  data=data,ax=ax[2,0])
sns.boxplot(x="y", y="emp.var.rate",  data=data,ax=ax[2,1])
sns.boxplot(x="y", y="cons.price.idx",  data=data,ax=ax[3,0])
sns.boxplot(x="y", y="cons.conf.idx", data=data,ax=ax[3,1])
sns.boxplot(x="y", y="euribor3m",  data=data,ax=ax[4,0])
sns.boxplot(x="y", y="nr.employed",  data=data,ax=ax[4,1])

plt.savefig('num vs target.png')

###categorical variables
fig, ax = plt.subplots(5, 2,figsize=(20,20))

sns.countplot(x='y', hue='job',data=data,palette="pastel",ax=ax[0,0])
sns.countplot(x='y', hue='marital',data=data,palette="pastel",ax=ax[0,1])
sns.countplot(x='y', hue='education',data=data,palette="pastel",ax=ax[1,0])
sns.countplot(x='y', hue='default',data=data,palette="pastel",ax=ax[1,1])
sns.countplot(x='y', hue='housing',data=data,palette="pastel",ax=ax[2,0])
sns.countplot(x='y', hue='loan',data=data,palette="pastel",ax=ax[2,1])
sns.countplot(x='y', hue='contact',data=data,palette="pastel",ax=ax[3,0])
sns.countplot(x='y', hue='month',data=data,palette="pastel",ax=ax[3,1])
sns.countplot(x='y', hue='day_of_week',data=data,palette="pastel",ax=ax[4,0])
sns.countplot(x='y', hue='poutcome',data=data,palette="pastel",ax=ax[4,1])

plt.savefig('cat vs target.png')


###correlations
numdata=data.filter(items=[
'age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y'])

plt.figure(figsize = (10,10))
corrMatrix = numdata.corr()
sns.heatmap(corrMatrix, annot=True)
plt.savefig('correlation.png')


#%%%%%
#CLEANING DATA

#replace target variable to binary
data['y'].replace('no', 0, inplace=True)
data['y'].replace('yes', 1, inplace=True)

#replace categorical unknown value with the mode
data.job.replace({'unknown':'admin.'}, inplace=True)
data.marital.replace({'unknown':'married'}, inplace=True)
data.education.replace({'unknown':'university.degree'}, inplace=True)
data.default.replace({'unknown':'no'}, inplace=True)
data.housing.replace({'unknown':'yes'}, inplace=True)
data.loan.replace({'unknown':'no'}, inplace=True)

#replace inconsistent data 
data['pdays'].mask((data['pdays']==999) & (data['previous']>0) & (data['poutcome']=='failure'), round(data['pdays'][(data['pdays'] != 999)].mean(),0), inplace=True)

#binning and trimming
data.education.replace({'illiterate':'basic.4y'}, inplace=True)
data.education.replace({'basic.4y':'4yandless'}, inplace=True)

data['pdays_bin']=pd.cut(x = data['pdays'],
                        bins=[-1,1,3,5,998,999], 
                        labels = ['0-1day', '2-3days', '4-5days','6plus','notcontacted'])
data["pdays_bin"] = data['pdays_bin'].astype('object')
data['previous_bin']=pd.cut(x = data['previous'],
                        bins=[-1,0,1,2,3,100], 
                        labels = ['notcontacted', '1time', '2time','3time','4plustime'])
data["previous_bin"] = data['previous_bin'].astype('object')
#drop columns
data.drop(columns=['emp.var.rate', 'euribor3m','duration','ModelPrediction','pdays','previous'], inplace=True)

#check skewedness then see if log transform helps
#want skew between -1 and 1
data['age'].skew()
data['campaign'].skew()
data["campaign"] = data["campaign"].map(lambda i: np.log(i) if i > 0 else 0) 
data['campaign'].skew()

#remove outliers based on percentiles
data['age'].describe()
Lower=data['age'].quantile(0.01)
Upper=data['age'].quantile(0.99)
data["age"] = np.where(data["age"] <Lower, Lower,data['age'])
data["age"] = np.where(data["age"] >Upper, Upper,data['age'])
data['age'].describe()

data['campaign'].describe()
Lower=data['campaign'].quantile(0.01)
Upper=data['campaign'].quantile(0.99)
data["campaign"] = np.where(data["campaign"] <Lower, Lower,data['campaign'])
data["campaign"] = np.where(data["campaign"] >Upper, Upper,data['campaign'])
data['campaign'].describe()

data['cons.conf.idx'].describe()
Lower=data['cons.conf.idx'].quantile(0.01)
Upper=data['cons.conf.idx'].quantile(0.99)
data["cons.conf.idx"] = np.where(data["cons.conf.idx"] <Lower, Lower,data['cons.conf.idx'])
data["cons.conf.idx"] = np.where(data["cons.conf.idx"] >Upper, Upper,data['cons.conf.idx'])
data['cons.conf.idx'].describe()

##hot encoding categorical
for column in data.columns:
 if data[column].dtype==object:
  dummyCols=pd.get_dummies(data[column], prefix=column)
  data=data.join(dummyCols)
  del data[column]

###SPLIT THE DATA TO X AND Y

X=data.drop('y',axis=1)
Y=data['y']

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=27)

#normalize continuous
num_d = X_train.select_dtypes(exclude=['object','uint8'])
num_columns=num_d.columns
for i in num_columns:
    
    # fit on training data column
    scale = StandardScaler().fit(X_train[[i]])
    
    # transform the training data column
    X_train[i] = scale.transform(X_train[[i]])
    
    # transform the testing data column
    X_test[i] = scale.transform(X_test[[i]])



#%%%%%
#EVALUATE RESULT OF CURRENT MODEL

#retrieve the model predictions and real labels
yprob=data_orig['ModelPrediction']
yreal=data_orig['y']
yreal.replace('no', 0, inplace=True)
yreal.replace('yes', 1, inplace=True)


#look at a graph of the model prediction values
sns.histplot(data=data_orig, x="ModelPrediction" , hue='y')
plt.savefig('prediction hist.png')

#ROC curve
auc_roc = round(roc_auc_score(yreal, yprob),3)
#wnant are close to 1
fpr, tpr, thresholds = roc_curve(yreal, yprob)
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC : AUC='+str(auc_roc))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
        
plot_roc_curve(fpr, tpr)
plt.savefig('roc curve.png')  

#precision vs recall
precision, recall, thresholds = precision_recall_curve(yreal, yprob)
no_skill = len(yreal[yreal==1]) / len(yreal)
plt.figure(figsize = (10,8))
plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
plt.plot(recall, precision, label = 'Knn')
plt.xlabel('recall')
plt.ylabel('precision')

#%%%%%
#CREATE A NEW MODEL

# load library
rfc = RandomForestClassifier()

# fit the predictor and target
rfc.fit(X_train, y_train)

# predict
rfc_predict = rfc.predict(X_test)# check performance

#evaluate performance
rfc_roc = metrics.plot_roc_curve(rfc, X_test, y_test)
rfc_pr=metrics.plot_precision_recall_curve(rfc, X_test, y_test)


# Retrieve and sort the feature importance in descending order
importances = rfc.feature_importances_

sorted_indices = np.argsort(importances)[::-1]

#Visualize the feature importance
plt.figure(figsize = (10,8))
plt.title('Feature Importance')
plt.bar(range(X_test.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_test.shape[1]), X_test.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.savefig('Importance.png')  



