#!/usr/bin/env python
# coding: utf-8

# In[6]:


from numpy import argmax
# define input string
data = 'priyadharshini'
print(data)


# In[3]:


# define universe of possible input values
alphabet = 'abcdefghijklmnopqrstuvwxyz '


# In[4]:


# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))


# In[7]:


# integer encode input data
integer_encoded = [char_to_int[char] for char in data]
print(integer_encoded)


# In[8]:


# one hot encode
onehot_encoded = list()
for value in integer_encoded:
	letter = [0 for _ in range(len(alphabet))]
	letter[value] = 1
	onehot_encoded.append(letter)
print(onehot_encoded)


# In[9]:


# invert encoding
inverted = int_to_char[argmax(onehot_encoded[0])]
print(inverted)


# In[10]:


#onehot encoding with scikit learn
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = ['bitter', 'bitter', 'sour', 'bitter', 'sweet', 'sweet', 'sour', 'bitter', 'sour', 'sweet']
values = array(data)
print(values)


# In[11]:


# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)


# In[12]:


# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)


# In[13]:


# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)


# In[14]:


from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# define example
data = [2, 6, 8, 1, 9, 2, 5, 3, 8, 0]
data = array(data)
print(data)
# one hot encode
encoded = to_categorical(data)
print(encoded)
# invert encoding
inverted = argmax(encoded[0])
print(inverted)


# In[ ]:




