#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
 
arr = np.array([['M', 99.4, 45.2, 55, 4.1, 220000, 'no'],
               ['M', 73.2, 66.2, 80, 9.9, 90000, 'no'],
               ['F', 50.0, 83.2, 78, 3.4, 290000, 'yes'],
               ['F', 89.4, 65.2, 96, 5.9, 50000, 'yes'],
               ['M', 66.4, 87.2, 40, 5.11, 70000, 'no']])
#
# Create Pandas DataFrame
#
df = pd.DataFrame(arr)
df.columns = ['gender', 'hsc_p', 'ssc_p', 'age', 'height', 'salary', 'suffer_from_disease']
#
# Convert the string data type to int and float appropriately
#
df[['age', 'salary']] = df[['age', 'salary']].astype(int)
df[['ssc_p', 'hsc_p', 'height']] = df[['ssc_p', 'hsc_p', 'height']].astype(float)


# In[5]:


df.head()


# In[6]:


def normalize(values):
    return (values - values.min())/(values.max() - values.min())


# In[7]:


cols = ['hsc_p', 'ssc_p', 'age', 'height', 'salary']
#
# Normalize the feature columns
#
df[cols] = df[cols].apply(normalize)


# In[8]:


def standardize(values):
    return (values - values.mean())/values.std()


# In[9]:


cols = ['hsc_p', 'ssc_p', 'age', 'height', 'salary']
#
# Standardize the feature columns; Dataframe needs to be recreated
#
df[cols] = df[cols].apply(standardize)


# In[10]:


from sklearn.preprocessing import MinMaxScaler
 
mmscaler = MinMaxScaler()
cols = ['hsc_p', 'ssc_p', 'age', 'height', 'salary']
df[cols] = mmscaler.fit_transform(df[cols])


# In[12]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
 
iris = datasets.load_iris()
X = iris.data
y = iris.target
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
 
mmscaler = MinMaxScaler()
X_train_norm = mmscaler.fit_transform(X_train)
X_test_norm = mmscaler.transform(X_test)


# In[13]:


from sklearn.preprocessing import StandardScaler
 
sc = StandardScaler()
cols = ['hsc_p', 'ssc_p', 'age', 'height', 'salary']
df[cols] = sc.fit_transform(df[cols])


# In[15]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
iris = datasets.load_iris()
X = iris.data
y = iris.target
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
 
sc = StandardScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)


# In[ ]:




