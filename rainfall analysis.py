
# coding: utf-8

# In[199]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df1=pd.read_csv('datafile.csv')


# In[200]:


df1.head()


# In[201]:


df1.drop("ANN",axis=1,inplace=True)
df1.drop("Jan-Feb",axis=1,inplace=True)
df1.drop("Mar-May",axis=1,inplace=True)
df1.drop("Jun-Sep",axis=1,inplace=True)
df1.drop("Oct-Dec",axis=1,inplace=True)


# In[202]:


df1.columns = ['Year',1,2,3,4,5,6,7,8,9,10,11,12]


# In[203]:


df1.dropna(inplace=True)


# In[204]:


df = pd.melt(df1, id_vars=["Year",], var_name="Month", value_name="Rainfall")
df = df2.sort_values(["Year","Month"])


# In[205]:


df['Rainfall']=df['Rainfall'].round()


# In[206]:


train = df.loc[df['Year'] >= 2007]
test = df.loc[df['Year'] == 2013]


# In[207]:


X_train=train
X_train=X_train[X_train['Year'] != 2013]
Y_train=X_train['Rainfall']
X_train.drop('Rainfall',axis=1)


# In[208]:


Y_train


# In[209]:


X_test=test
Y_test=test['Rainfall']


# In[210]:


from sklearn import svm
from sklearn.svm import SVC
model = svm.SVC(gamma='auto',kernel='linear')
model.fit(X_train, Y_train)


# In[211]:


Y_pred = model.predict(X_test)


# In[212]:


df1 = pd.DataFrame({'Actual Rainfall': Y_test, 'Predicted Rainfall': Y_pred})  
df1[df1['Predicted Rainfall']!=0].head(10)


# In[213]:


Y_test1=test['Rainfall']


# In[215]:


plt.figure(figsize=(20,10))
plt.scatter(X_test['Month'],Y_test1,color='red')
plt.plot(X_test['Month'],Y_test,color='green')
plt.plot(X_test['Month'],Y_pred,color='blue')


# In[220]:


from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, Y_pred))


# In[222]:


print(metrics.accuracy_score(Y_pred, Y_test) * 100)


# In[173]:


from sklearn.svm import SVR
regressor=SVR(gamma='scale',kernel='linear')
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)


# In[174]:


df1 = pd.DataFrame({'Actual Rainfall': Y_test, 'Predicted Rainfall': Y_pred})  
df1[df1['Predicted Rainfall']>=1].head(10)


# In[176]:


plt.figure(figsize=(20,10))
plt.scatter(X_test['Month'],Y_test1,color='red')
plt.plot(X_test['Month'],Y_test,color='green')
plt.plot(X_test['Month'],Y_pred,color='blue')


# In[177]:


from sklearn.neighbors import KNeighborsClassifier
neig = KNeighborsClassifier(n_neighbors=1)
neig.fit(X_train,Y_train)


# In[178]:


Y_pred = neig.predict(X_test)


# In[179]:


df1 = pd.DataFrame({'Actual Rainfall': Y_test, 'Predicted Rainfall': Y_pred})  
df1[df1['Predicted Rainfall']>=1].head(10)


# In[181]:


plt.figure(figsize=(20,10))
plt.scatter(X_test['Month'],Y_test1,color='red')
plt.plot(X_test['Month'],Y_test,color='green')
plt.plot(X_test['Month'],Y_pred,color='blue')


# In[195]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,Y_train)


# In[196]:


Y_pred = mlp.predict(X_test)


# In[197]:


df1 = pd.DataFrame({'Actual Rainfall': Y_test, 'Predicted Rainfall': Y_pred})  
df1[df1['Predicted Rainfall']>=1].head(10)


# In[198]:


plt.figure(figsize=(20,10))
plt.scatter(X_test['Month'],Y_test1,color='red')
plt.plot(X_test['Month'],Y_test,color='green')
plt.plot(X_test['Month'],Y_pred,color='blue')

