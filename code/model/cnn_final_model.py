#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input, Dense, Dropout, Activation, LSTM, Lambda, Add, Subtract
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
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import tensorflow as tf
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def neg_sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return 1 - true_positives / (possible_positives + K.epsilon())

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# In[2]:


train_set = np.load('train_set_score.npy')


# In[3]:


test_set_balanced = np.load('test_set_score.npy')


# In[5]:


print(train_set.shape)
print(test_set_balanced.shape)

test_set_bin = np.vectorize(lambda x: int(x))(test_set[:, 2] > 0.5)
#print(np.sum(test_set_bin))
idx_1 = np.where(test_set_bin == 1)
idx_1 = np.array(idx_1).reshape(len(idx_1[0]),)
idx_0 = np.where(test_set_bin == 0)
idx_0 = np.random.choice(np.array(idx_0).reshape(len(idx_0[0]),), np.sum(test_set_bin), replace = False)
test_set_balanced = test_set[np.append(idx_1, idx_0), :]
print(test_set_balanced.shape)

train_set_bin = np.vectorize(lambda x: int(x))(train_set[:, 2] > 0.5)
#print(np.sum(train_set_bin))
idx_1 = np.where(train_set_bin == 1)
idx_1 = np.array(idx_1).reshape(len(idx_1[0]),)
idx_0 = np.where(train_set_bin == 0)
idx_0 = np.random.choice(np.array(idx_0).reshape(len(idx_0[0]),), np.sum(train_set_bin), replace = False)
train_set_balanced = train_set[np.append(idx_1, idx_0), :]
print(train_set_balanced.shape)


# In[6]:


np.random.shuffle(train_set_balanced)
np.random.shuffle(test_set_balanced)


# In[7]:


n_steps = 20 #10seconds
n_lap = 10 #5seconds

train_x = []
train_y = []

for i in range(10000):
    image1 = train_set_balanced[i][0][-40:,]
    image2 = train_set_balanced[i][1][-40:,]
    y = int(train_set_balanced[i][2] > 0.5)
    #y = train_set[i][2]
    train_x.append(np.array([image1, image2]))
    train_y.append(y)
    
train_x = np.array(train_x)
print(train_x.shape)
train_y = np.array(train_y)
print(train_y.shape)

validation_x = []
validation_y = []
for i in range(10000, len(train_set_balanced)):
    image1 = train_set_balanced[i][0][-40:,]
    image2 = train_set_balanced[i][1][-40:,]
    y = int(train_set_balanced[i][2] > 0.5)
    #y = train_set[i][2]
    validation_x.append(np.array([image1, image2]))
    validation_y.append(y)
    
validation_x = np.array(validation_x)
print(validation_x.shape)
validation_y = np.array(validation_y)
print(validation_y.shape)

test_x = []
test_y = []
for i in range(len(test_set_balanced)):
    image1 = test_set_balanced[i][0][-40:,]
    image2 = test_set_balanced[i][1][-40:,]
    y = int(test_set_balanced[i][2] > 0.5)
    #y = val_set[i][2]
    test_x.append(np.array([image1, image2]))
    test_y.append(y)
    
test_x = np.array(test_x)
print(test_x.shape)
test_y = np.array(test_y)
print(test_y.shape)


def get_model():
    kernel_size = 5
    filters = 5
    stride_width = 1
    stirde_height = 3
    pool_size = 4

    inp = Input(shape = (2, 40, 120, 3))
    
    mat1 = Lambda(lambda x: x[:, 0])(inp)
    mat2 = Lambda(lambda x: x[:, 1])(inp)

    mat1 = Convolution2D(filters = 5, kernel_size = (5,10), 
                                         strides = (1, 1), padding = "same", activation = 'softmax')(mat1)
    mat2 = Convolution2D(filters = 5, kernel_size = (5,10), 
                                         strides = (1, 1), padding = "same", activation = 'softmax')(mat2)
    
    #mat1 = Dropout(rate = 0.25)(mat1)
    #mat2 = Dropout(rate = 0.25)(mat2)
    mat1 = MaxPooling2D(pool_size = 4, strides = 2)(mat1)
    mat2 = MaxPooling2D(pool_size = 4, strides = 2)(mat2)
    mat1 = Convolution2D(filters = 3, kernel_size = 2, 
                                         strides = (1, 1), padding = "same", activation = 'softmax')(mat1)
    mat2 = Convolution2D(filters = 3, kernel_size = 2, 
                                         strides = (1, 1), padding = "same", activation = 'softmax')(mat2)
    
    mat1 = MaxPooling2D(pool_size = 2, strides = 2)(mat1)
    mat2 = MaxPooling2D(pool_size = 2, strides = 2)(mat2)
#     mat1 = Dropout(rate = 0.25)(mat1)
#     mat2 = Dropout(rate = 0.25)(mat2)
    mat1 = Convolution2D(filters = 3, kernel_size = 2, 
                                         strides = (1, 1), padding = "same", activation = 'softmax')(mat1)
    mat2 = Convolution2D(filters = 3, kernel_size = 2, 
                                         strides = (1, 1), padding = "same", activation = 'softmax')(mat2)
    mat1 = MaxPooling2D(pool_size = 2, strides = 2)(mat1)
    mat2 = MaxPooling2D(pool_size = 2, strides = 2)(mat2)
    mat1 = Flatten()(mat1)
    mat2 = Flatten()(mat2)

    vec3 = dot([mat1, mat2], axes = 1)
    vec4 = concatenate([mat1, mat2, vec3])
    vec3 = Add()([mat1, mat2])
    vec4 = concatenate([vec4, vec3])
    vec3 = Subtract()([mat1, mat2])
    vec4 = concatenate([vec4, vec3])
    
    vec4 = Dense(256,  activation='sigmoid')(vec4)
   # vec4 = Dropout(rate=0.25)(vec4)
    vec4 = Dense(30,  activation='relu')(vec4)
   
    preds = Dense(1, activation='sigmoid')(vec4)
    
    model = Model(inputs=inp, outputs=preds)
    
    opt = Adam(lr=0.0001, decay=5*1e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics =['acc'])
    
    return model
    
get_model()


# In[ ]:


model = get_model()
early_stopping = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience = 90)

history = model.fit(train_x, train_y, validation_data=(validation_x, validation_y), 
    epochs=200, batch_size=32,shuffle=True)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

preds_val = model.predict(validation_x)
val_acc = accuracy_score(preds_val > 0.5, validation_y > 0.5)
fpr, tpr, thresholds = roc_curve(validation_y > 0.5, preds_val > 0.5)
conf_mat = confusion_matrix(validation_y > 0.5, preds_val > 0.5)
print(val_acc)
print(conf_mat)
plt.plot(fpr, tpr, marker = '.')
plt.show()

preds_test = model.predict(test_x)
test_acc = accuracy_score(preds_test > 0.5, test_y > 0.5)
fpr, tpr, thresholds = roc_curve(test_y > 0.5, preds_test > 0.5)
conf_mat = confusion_matrix(test_y > 0.5, preds_test > 0.5)
print(test_acc)
print(conf_mat)
plt.plot(fpr, tpr, marker = '.')
plt.show()


#



