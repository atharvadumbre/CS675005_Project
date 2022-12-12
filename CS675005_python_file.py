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


# In[ ]:


sns.pairplot(df, hue="Fire Alarm", kind= 'kde',corner=True)


# In[ ]:


sns.displot(df, x='Temperature')


# In[ ]:


sns.displot(df, x='Humidity')


# In[ ]:


sns.displot(df, x='TVOC')


# In[ ]:


sns.displot(df, x='Raw H2')


# In[ ]:


sns.displot(df, x='Raw Ethanol'


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


# In[44]:


model_df.sort_values(by = 'Accuracy',ascending = False,inplace = True)
plt.figure(figsize=(25,10))
plt.plot(model_df['Name'], model_df['Accuracy']) 
plt.title('Accuracy VS Model')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig('accuracy_model.png')
plt.show()


# In[45]:


model_df.sort_values(by = 'F1 Score',ascending = False,inplace = True)
plt.figure(figsize=(25,10))
plt.plot(model_df['Name'], model_df['F1 Score']) 
plt.title('F1 Score VS Model')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.grid()
plt.savefig('f1score_model.png')
plt.show()


# In[46]:


model_df.sort_values(by = 'AUC',ascending = False,inplace = True)
plt.figure(figsize=(25,10))
plt.plot(model_df['Name'], model_df['AUC']) 
plt.title('AUC VS Model')
plt.xlabel('Models')
plt.ylabel('AUC')
plt.grid()
plt.savefig('auc_model.png')
plt.show()


# In[ ]:


model0 = LogisticRegression()
model0.fit(X_train_smote,y_train_smote)
prediction = model0.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = model0.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
scores = cross_val_score(model0, X_train_smote, y_train_smote, cv=5)
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


# In[ ]:


model1 = GaussianNB()
model1.fit(X_train_smote,y_train_smote)
prediction = model1.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = model1.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print(f'Accuracy: {accuracyScore}')
print(f'F1 Score: {f1score}')
print(f'AUC: {auc}')
scores = cross_val_score(model1, X_train_smote, y_train_smote, cv=5)

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


# In[ ]:


model2 = SGDClassifier(loss = 'log')
model2.fit(X_train_smote,y_train_smote)
prediction = model2.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = model2.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
scores = cross_val_score(model2, X_train_smote, y_train_smote, cv=5)
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


# In[ ]:


model3 = KNeighborsClassifier()
model3.fit(X_train_smote,y_train_smote)
prediction = model3.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = model3.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
scores = cross_val_score(model3, X_train_smote, y_train_smote, cv=5)
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


# In[ ]:


model4 = DecisionTreeClassifier()
model4.fit(X_train_smote,y_train_smote)
prediction = model4.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = model4.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
scores = cross_val_score(model4, X_train_smote, y_train_smote, cv=5)
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


# In[ ]:


model5 = RandomForestClassifier()
model5.fit(X_train_smote,y_train_smote)
prediction = model5.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = model5.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
scores = cross_val_score(model5, X_train_smote, y_train_smote, cv=5)
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


# In[ ]:


model6 = GradientBoostingClassifier()
model6.fit(X_train_smote,y_train_smote)
prediction = model6.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = model6.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
scores = cross_val_score(model6, X_train_smote, y_train_smote, cv=5)
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


# In[ ]:


model7 =  AdaBoostClassifier()
model7.fit(X_train_smote,y_train_smote)
prediction = model7.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = model7.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
scores = cross_val_score(model7, X_train_smote, y_train_smote, cv=5)
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


# In[ ]:


model8 =  LGBMClassifier()
model8.fit(X_train_smote,y_train_smote)
prediction = model8.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = model8.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
scores = cross_val_score(model8, X_train_smote, y_train_smote, cv=5)
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


# In[ ]:


model9 =  XGBClassifier()
model9.fit(X_train_smote,y_train_smote)
prediction = model9.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = model9.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
scores = cross_val_score(model9, X_train_smote, y_train_smote, cv=5)
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


# In[ ]:


model10 =  LinearDiscriminantAnalysis()
model10.fit(X_train_smote,y_train_smote)
prediction = model10.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = model10.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
scores = cross_val_score(model10, X_train_smote, y_train_smote, cv=5)
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


# In[ ]:


model11 =  MLPClassifier(alpha=1, max_iter=1000)
model11.fit(X_train_smote,y_train_smote)
prediction = model11.predict(X_test)
accuracyScore = accuracy_score(prediction,y_test)
f1score = f1_score(prediction,y_test)
probs = model11.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
scores = cross_val_score(model11, X_train_smote, y_train_smote, cv=5)
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


target_names = ['Not Fire','Fire']
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

