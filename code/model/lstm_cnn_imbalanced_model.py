#!/usr/bin/env python
# coding: utf-8

# In[175]:


# multivariate data preparation
import numpy as np
from numpy import array
from numpy import hstack
import os
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps, n_lap):
	X = list()
	for i in range(0, len(sequences), n_lap):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x = sequences[i:end_ix, :]
		X.append(seq_x)
		
	return array(X)


# In[176]:


from keras.layers import Input, Dense, Dropout, Activation, LSTM, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.initializers import RandomUniform
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Reshape
from keras.layers.merge import concatenate, dot
from keras.layers.core import Reshape
from keras import backend as K
from sklearn.metrics import accuracy_score


# In[177]:


def get_model():
    kernel_size = 5
    filters = 10 
    stride = 4
    pool_size = 4
    lstm_output_size = 100
    
    inp = Input(shape = (2, 11, 20, 64, 3))
    mat1 = Lambda(lambda x: x[:, 0])(inp)
    mat2 = Lambda(lambda x: x[:, 1])(inp)
    mat1 = TimeDistributed(Convolution2D(filters = filters, kernel_size = kernel_size, 
                                         strides = (stride, stride), activation = 'relu'))(mat1)
    mat2 = TimeDistributed(Convolution2D(filters = filters, kernel_size = kernel_size, 
                                         strides = (stride, stride), activation = 'relu'))(mat2)
    
    mat1 = TimeDistributed(MaxPooling2D(pool_size = pool_size, strides = 2))(mat1)
    mat2 = TimeDistributed(MaxPooling2D(pool_size = pool_size, strides = 2))(mat2)
    mat1 = TimeDistributed(Dropout(rate = 0.25))(mat1)
    mat2 = TimeDistributed(Dropout(rate = 0.25))(mat2)
    mat1 = TimeDistributed(Flatten())(mat1)
    mat2 = TimeDistributed(Flatten())(mat2)
    
    mat1 = LSTM(lstm_output_size)(mat1)
    mat2 = LSTM(lstm_output_size)(mat2)
    
    vec3 = dot([mat1, mat2], axes = 1)
    vec4 = concatenate([mat1, mat2, vec3])

    vec4 = Dense(100,  activation='relu')(vec4)
    vec4 = Dropout(rate=0.25)(vec4)
    preds = Dense(1, activation='sigmoid')(vec4)
    
    model = Model(inputs=inp, outputs=preds)
    
    opt = RMSprop(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    
    return model
    
get_model()
    
    


# In[178]:


train_set = np.load('train_set.npy')

train = train_set[:5000, :]
validation = train_set[5000:, :]

train_x = train[:, :2]
train_y = train[:, 2]
validation_x = test[:, :2]
validation_y = test[:, :2]

    
    


# In[179]:


n_steps = 20 #10seconds
n_lap = 10 #5seconds

train_x = []
train_y = []
for i in range(6000):
    image1 = split_sequences(np.transpose(train_set[i][0], (1,0,2)), n_steps, n_lap)
    image2 = split_sequences(np.transpose(train_set[i][1], (1,0,2)), n_steps, n_lap)
    y = train_set[i][2]
    train_x.append(np.array([image1, image2]))
    train_y.append(y)
    
train_x = np.array(train_x)
print(train_x.shape)
train_y = np.array(train_y)
print(train_y.shape)

validation_x = []
validation_y = []
for i in range(6000, 7400):
    image1 = split_sequences(np.transpose(train_set[i][0], (1,0,2)), n_steps, n_lap)
    image2 = split_sequences(np.transpose(train_set[i][1], (1,0,2)), n_steps, n_lap)
    y = train_set[i][2]
    validation_x.append(np.array([image1, image2]))
    validation_y.append(y)
    
validation_x = np.array(validation_x)
print(validation_x.shape)
validation_y = np.array(validation_y)
print(validation_y.shape)


# In[180]:


model = get_model()
early_stopping = EarlyStopping(monitor='val_acc', patience=5)

model.fit(train_x, train_y, validation_data=(validation_x, validation_y), 
    epochs=100, batch_size=100, shuffle=True, callbacks=[early_stopping])

preds_val = model.predict(validation_x)
val_auc = accuracy_score(preds_val > 0.5, validation_y)


# In[ ]:




