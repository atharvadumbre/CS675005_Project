#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('smoke_detection_iot.csv')
df.head(5)


# In[3]:


df.describe()


# In[4]:


pd.isnull(df).sum()


# In[5]:


df = df.drop(columns=['Unnamed: 0','UTC','CNT'])
df.head()


# In[6]:


df.rename(columns = {'Temperature[C]':'Temperature', 'Humidity[%]':'Humidity','TVOC[ppb]':'TVOC', 'eCO2[ppm]':'eCO2','Pressure[hPa]':'Pressure'}, inplace = True)


# In[8]:


plt.figure(figsize = (10,10))
sns.heatmap(df.corr(),annot = True)
plt.savefig('correlation_matrix.png')


# In[9]:


cor_matrix = df.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(to_drop)


# In[10]:


df1 = df.drop(columns= to_drop, axis=0)
df1.head()


# In[11]:


plt.hist(df['Fire Alarm'])
plt.xlabel('Fire or Not Fire')
plt.ylabel('Count of Samples')
plt.savefig('target_bar.png')


# In[12]:


plt.pie(df['Fire Alarm'].value_counts(),labels=['Fire','No Fire'],autopct='%1.2f%%',colors=['green','yellow'])
plt.title('Fire Alarm')
plt.savefig('target_pie.png')
plt.show()


# In[13]:


df['Fire Alarm'].value_counts()


# In[15]:


X = df1.drop('Fire Alarm', axis=1)
y = df1['Fire Alarm']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
print(f'X_train shape:{X_train.shape} and y_trian shape:{y_train.shape}')
print(f'X_test shape:{X_test.shape} and y_test shape:{y_test.shape}')


# In[16]:


sns.countplot(x = y_train)
plt.savefig('before_smote.png')


# In[19]:


smote = SMOTE(sampling_strategy='not majority')
X_train_smote,y_train_smote = smote.fit_resample(X_train,y_train)
sns.countplot(x=y_train_smote)
plt.savefig('after_smote.png')


# In[21]:


models = [LogisticRegression(), GaussianNB(), SGDClassifier(loss = 'log'), KNeighborsClassifier(), DecisionTreeClassifier(),
         RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(), LGBMClassifier(), XGBClassifier(), LinearDiscriminantAnalysis(),
         MLPClassifier(alpha=1, max_iter=1000)]


# In[22]:


Name = []
Accuracy = []
F1 = []
AUC = []
for model in models:
    Name.append(type(model).__name__)
    model.fit(X_train_smote,y_train_smote)
    prediction = model.predict(X_test)
    accuracyScore = accuracy_score(prediction,y_test)
    f1score = f1_score(prediction,y_test)
    probs = model.predict_proba(X_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=10)
    Accuracy.append(accuracyScore)
    F1.append(f1score)
    AUC.append(auc)
    
Dict = {'Name':Name,'Accuracy':Accuracy,'F1 Score':F1,'AUC': AUC}
model_df = pd.DataFrame(Dict)
model_df


# In[37]:


estimators = [('log',LogisticRegression()),
              ('rf',RandomForestClassifier()),
              ('lgbm',LGBMClassifier()),
              ('xgb',XGBClassifier()),
              ('mlp',MLPClassifier(alpha=1, max_iter=1000)),
              ('gauss',GaussianNB())
            ]

stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack.fit(X_train_smote, y_train_smote).score(X_test, y_test)


# In[ ]:


prediction = stack.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = stack.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
scores = cross_val_score(stack, X_train, y_train, cv=5)
print(f'Accuracy: {accuracyScore}')
print(f'F1 Score: {f1score}')
print(f'AUC: {auc}')
print("CV Score: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print('Classification Report: ')
print(classification_report(y_test, prediction, target_names=target_names))
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
cf_matrix = confusion_matrix(y_test, prediction)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

