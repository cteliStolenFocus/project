#!/usr/bin/env python
# coding: utf-8

# #### AAI 521 - <b><font color='Red'>Final Project</font></b>
# - This module author: <b><font color='Red'>PRAKASH PERIMBETI</font></b>
# - TEAM: Christopher Teli (Lead), Olympia Saha and Prakash Perimbeti
# - Date: December 12, 2022
# - Professor: Dr. Saeed Sardari

# In[1]:


import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from time import perf_counter
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.layers as tfl
from keras.optimizers import SGD   
from  sklearn.preprocessing import LabelEncoder 


# In[34]:


#Get image file names and its classifiers
trainimage_dir = Path('./train')
trainlabels = []
# Get filepaths and labels
#filepaths =  os.listdir(image_dir)
Tfilepaths = list(trainimage_dir.glob(r'**/*.jpg'))
count = 0 

for filename in filepaths:
    lbl = str(filename)[6:7]
    trainlabels.append(lbl)
    count+=1

print(len(Tfilepaths), len(trainlabels))
print(Tfilepaths[4000], trainlabels[4000])


# total 2515 images
filepaths = pd.Series(Tfilepaths, name='Filepath').astype(str)
labels = pd.Series(trainlabels, name='Label')

# Concatenate filepaths and labels
T_IMAGE_DF = pd.concat([filepaths, labels], axis=1)

# Shuffle the DataFrame and reset index
T_IMAGE_DF = T_IMAGE_DF.sample(frac=1).reset_index(drop = True)

#le_Label_ACTIVE = LabelEncoder()
#IMAGE_DF['Label'] = le_Label_ACTIVE.fit_transform(IMAGE_DF['Label'])

#IMAGE_DF['Label'] = pd.Categorical(IMAGE_DF.Label)
print(T_IMAGE_DF.shape)
# Show the result
T_IMAGE_DF.head(5)
# Separate in train and test data
T_TRAIN_DF, T_TEST_DF = train_test_split(T_IMAGE_DF, train_size=0.9, shuffle=True, random_state=1)
print(T_IMAGE_DF.shape, T_TRAIN_DF.shape, T_TEST_DF.shape)
# Create the generators
T_train_generator,T_test_generator,T_TRAIN_IMAGES,T_VAL_IMAGES,T_TEST_IMAGES=create_gen(T_TRAIN_DF, T_TEST_DF)
print('\n')


# In[ ]:


#Get image file names and its classifiers
image_dir = Path('./imagedata')
labels = []
# Get filepaths and labels
#filepaths =  os.listdir(image_dir)
filepaths = list(image_dir.glob(r'**/*.png'))
count = 0 
for filename in filepaths:
    labels.append(str(filename[6]))
    filepaths[count] = './imagedata' + '/' + filename
    count+=1
    


# In[6]:


from  sklearn.preprocessing import LabelEncoder 
# total 2515 images
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
IMAGE_DF = pd.concat([filepaths, labels], axis=1)

# Shuffle the DataFrame and reset index
IMAGE_DF = IMAGE_DF.sample(frac=1).reset_index(drop = True)

#le_Label_ACTIVE = LabelEncoder()
#IMAGE_DF['Label'] = le_Label_ACTIVE.fit_transform(IMAGE_DF['Label'])

#IMAGE_DF['Label'] = pd.Categorical(IMAGE_DF.Label)
print(IMAGE_DF.shape)
# Show the result
IMAGE_DF.head(5)


# In[7]:


print(len(filepaths), len(labels), labels[4])


# In[8]:


# Separate in train and test data
TRAIN_DF, TEST_DF = train_test_split(IMAGE_DF, train_size=0.9, shuffle=True, random_state=1)
print(IMAGE_DF.shape, IMAGE_DF.shape, IMAGE_DF.shape)


# In[9]:


#8586586550
#
# This loads images using Image Data Generator
def create_gen(train_df, test_df):
    # Load the Images with a generator and Data Augmentation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training',
        rotation_range=30, # Uncomment to use data augmentation
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='validation',
        rotation_range=30, # Uncomment to use data augmentation
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    return train_generator,test_generator,train_images,val_images,test_images


# In[10]:


# Create the generators
train_generator,test_generator,TRAIN_IMAGES,VAL_IMAGES,TEST_IMAGES=create_gen(TRAIN_DF, TEST_DF)

print('\n')


# In[ ]:


saved = buildTransferModels(TRAIN_IMAGES, VAL_IMAGES)
print( "A total of :", saved," models from this build")


# In[10]:


sgd = SGD(learning_rate=0.001)
def get_model(model):
# Load the pretained model
    kwargs =    {'input_shape':(224, 224, 3),
                'include_top':False,
                'weights':'imagenet',
                'pooling':'avg'}
    
    pretrained_model = model(**kwargs)
    pretrained_model.trainable = False
    
    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(36, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=sgd,#'adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# In[38]:


# Dictionary with the models
models = {
    "InceptionResNetV2": {"model":tf.keras.applications.InceptionResNetV2, "perf":0},
    "ResNet101": {"model":tf.keras.applications.ResNet101, "perf":0},
    "InceptionV3": {"model":tf.keras.applications.InceptionV3, "perf":0},
    "DenseNet121": {"model":tf.keras.applications.DenseNet121, "perf":0},
    "MobileNet": {"model":tf.keras.applications.MobileNet, "perf":0},
    "VGG16": {"model":tf.keras.applications.VGG16, "perf":0},
}

def buildTransferModels(TRAIN_IMAGES, VAL_IMAGES):

  # Fit the models
  takeNext = False
  savedModels = 0
  for name, model in models.items():
    print("processing:", name)
    # Get the model
    m = get_model(model['model'])
    models[name]['model'] = m
    start = perf_counter()
    print('fitting the model:', name)
    # Fit the model
    try:
        history = m.fit(TRAIN_IMAGES,validation_data=VAL_IMAGES,epochs=10,verbose=1) 
    except Exception:
        print(Exception)
        print("Failed with:",name," Transfer learnning")
        takeNext = True
    if(takeNext is False):
        # Sav the duration, the train_accuracy and the val_accuracy
        duration = perf_counter() - start
        duration = round(duration,2)
        models[name]['perf'] = duration
        print(f"{name:20} trained in {duration} sec")
    
        val_acc = history.history['val_accuracy']
        models[name]['val_acc'] = [round(v,4) for v in val_acc]
    
        train_acc = history.history['accuracy']
        models[name]['train_accuracy'] = [round(v,4) for v in train_acc]
        m.save(name+".h5")
        savedModels +=1
  return savedModels


# In[11]:


#Get image file names and its classifiers
image_dir = Path('./imagedata')
labels = []
# Get filepaths and labels
filepaths =  os.listdir(image_dir)
#filepaths = list(image_dir.glob(r'**/*.png'))
count = 0 

for filename in filepaths:
    lbl = str(filename)[6:7]
    labels.append(lbl)
    filepaths[count] = './imagedata'+'/'+filename
    count+=1

print(len(filepaths), len(labels))
print(filepaths[400], labels[400])


# total 2515 images
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
IMAGE_DF = pd.concat([filepaths, labels], axis=1)

# Shuffle the DataFrame and reset index
IMAGE_DF = IMAGE_DF.sample(frac=1).reset_index(drop = True)

#le_Label_ACTIVE = LabelEncoder()
#IMAGE_DF['Label'] = le_Label_ACTIVE.fit_transform(IMAGE_DF['Label'])

#IMAGE_DF['Label'] = pd.Categorical(IMAGE_DF.Label)
print(IMAGE_DF.shape)
# Show the result
IMAGE_DF.head(5)
# Separate in train and test data
TRAIN_DF, TEST_DF = train_test_split(IMAGE_DF, train_size=0.9, shuffle=True, random_state=1)
print(IMAGE_DF.shape, TRAIN_DF.shape, TEST_DF.shape)
# Create the generators
train_generator,test_generator,TRAIN_IMAGES,VAL_IMAGES,TEST_IMAGES=create_gen(TRAIN_DF, TEST_DF)
print('\n')


# In[12]:


# Dictionary with the models
models = {
    "InceptionResNetV2": {"model":tf.keras.applications.InceptionResNetV2, "perf":0},
    "ResNet101": {"model":tf.keras.applications.ResNet101, "perf":0},
    "InceptionV3": {"model":tf.keras.applications.InceptionV3, "perf":0},
    "DenseNet121": {"model":tf.keras.applications.DenseNet121, "perf":0},
    "MobileNet": {"model":tf.keras.applications.MobileNet, "perf":0},
    "VGG16": {"model":tf.keras.applications.VGG16, "perf":0},
}

def buildTransferModels(TRAIN_IMAGES, VAL_IMAGES, name):

    model = models.get(name)
    print("processing:", name)
    # Get the model
    m = get_model(model['model'])
    models[name]['model'] = m
    start = perf_counter()
    print('fitting the model:', name)
    history = m.fit(TRAIN_IMAGES,
                    validation_data=VAL_IMAGES,
                    epochs=10,
                    verbose=1,
                   batch_size = 32)

    # Sav the duration, the train_accuracy and the val_accuracy
    duration = perf_counter() - start
    duration = round(duration,2)
    models[name]['perf'] = duration
    print(f"{name:20} trained in {duration} sec")
    
    val_acc = history.history['val_accuracy']
    models[name]['val_acc'] = [round(v,4) for v in val_acc]
    
    train_acc = history.history['accuracy']
    models[name]['train_accuracy'] = [round(v,4) for v in train_acc]
    m.save(name+"36.h5")


# In[29]:


buildTransferModels(TRAIN_IMAGES,VAL_IMAGES, "InceptionResNetV2")


# In[7]:


buildTransferModels(TRAIN_IMAGES,VAL_IMAGES, "ResNet101")


# In[ ]:


buildTransferModels(TRAIN_IMAGES,VAL_IMAGES, "InceptionV3")


# In[13]:


buildTransferModels(TRAIN_IMAGES,VAL_IMAGES, "DenseNet121")


# In[14]:


buildTransferModels(TRAIN_IMAGES,VAL_IMAGES, "MobileNet")


# In[15]:


buildTransferModels(TRAIN_IMAGES,VAL_IMAGES, "VGG16")


# In[34]:


m =  models["InceptionResNetV2"]['model']
ypredictionRN2 = m.predict(TEST_IMAGES)


# In[39]:


print(len(ypredictionRN2), ypredictionRN2[52])
def getSinglePredictedMaxColumn (prediction):
    num_rows, num_cols = prediction.shape
    print(num_rows, num_cols)
    new_list = []
    for i in range(num_rows):
        column_list = list(prediction[i])
        max_index = column_list.index(max(column_list))
        new_list.append(max_index)

    return(new_list)
newpred = getSinglePredictedMaxColumn(ypredictionRN2)
print(newpred[1:5])


# In[ ]:




