#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import exp


# In[2]:


x = [1.9,0.6,2.2,0.8]
y = [0,0,0,0]                                   #find exponents
soft_max_result = [0,0,0,0]


# In[3]:


for i in range(len(x)):
    y[i] = exp(x[i])
    print(x[i],":",y[i])                       #find exponents


# In[4]:


total_y = sum(y)
print("denominator ", total_y)
for i in range(len(x)):
    soft_max_result[i] = y[i]/total_y
print("softmax result")
print(soft_max_result)                               #normalize


# In[ ]:




