from keras.applications.vgg16 import VGG16  
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from sklearn.preprocessing import LabelEncoder, LabelBinarizer  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, TensorBoard,ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
import os
from google.colab.patches import cv2_imshow # for image display
!pip install pyyaml h5py  # Required to save models in HDF5 format
import h5py



trdata = ImageDataGenerator()
#traindata = trdata.flow_from_directory(directory="/content/drive/MyDrive/Colab/data1a/training",target_size=(224,224),classes=['Damaged', 'Not Damaged'],class_mode="sparse",batch_size=40,interpolation="nearest")
traindata = trdata.flow_from_directory("/content/drive/MyDrive/Colab/data1a/training",target_size=(224,224),batch_size=46,interpolation="nearest")
print(len(traindata))
print(traindata)

tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="/content/drive/MyDrive/Colab/data1a/validation",target_size=(224,224),batch_size=46,interpolation="nearest")
print(len(testdata))



model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

model.trainable = False

input = keras.Input(shape=(224,224,3))
# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(4096, activation='relu')(flat1)
class2 = Dense(4096, activation='relu')(class1)
output = Dense(2, activation='softmax')(class2)

# define new model
model = keras.Model(inputs=model.inputs, outputs=output)

model.summary()
