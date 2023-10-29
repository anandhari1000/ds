#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


# In[4]:


(X_train,y_train),(X_test,y_test)=mnist.load_data()


# In[5]:


X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
X_test=X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))


# In[6]:


print(X_train.shape)


# In[7]:


print(X_test.shape)


# In[8]:


X_train=X_train/255
X_test=X_test/255


# In[9]:


model=Sequential()


# In[11]:


model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))


# In[12]:


model.add(MaxPool2D(2,2))


# In[13]:


model.add(Flatten())
model.add(Dense(100,activation='relu'))


# In[14]:


model.add(Dense(100,activation='softmax'))


# In[15]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[16]:


model.fit(X_train,y_train,epochs=10)


# In[17]:


model.evaluate(X_test,y_test)


# In[19]:


model.evaluate(X_train,y_train)


# In[ ]:




