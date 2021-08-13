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




