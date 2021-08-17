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



# initialize our initial learning rate, # of epochs to train for,and batch size
INIT_LR = 0.0003      #0.0005
EPOCHS = 20
batch_size = 46  # or BS

# Checkpoints between the training steps
#model_checkpoint = ModelCheckpoint(filepath='/content/drive/MyDrive/Colab/VGG16/VGG_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',monitor='val_loss',
 #                                  verbose=1,save_best_only=True,save_weights_only=False,mode='auto',save_freq=20)

# Termination of training if the loss become Nan
terminate_on_nan = TerminateOnNaN()

# For watching the live loss, accuracy and graphs using tensorboard
#t_board = TensorBoard(log_dir='./logs', histogram_freq=0,batch_size=32, write_graph=True,write_grads=False,write_images=False, 
#                      embeddings_freq=0, update_freq='epoch')
                                

# For reducing the loss when loss hits a plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=0.0000001)

# combine all the call backs to feed to the model
#callbacks = [model_checkpoint, t_board, terminate_on_nan, reduce_lr]
#callbacks = [model_checkpoint, terminate_on_nan, reduce_lr]

# initialize the model and optimizers
opt = Adam(learning_rate=INIT_LR, beta_1=0.9, beta_2=0.999, amsgrad=False)

# compile the model with loss function, optimizer and the evaluating metrics
#model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
model.compile(loss="binary_crossentropy", optimizer= opt ,metrics=["accuracy"])




#checkpoint = ModelCheckpoint(filepath="/content/drive/MyDrive/Colab/VGG16/vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)

#checkpoint = ModelCheckpoint(filepath="/content/drive/MyDrive/Colab/VGG16", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

checkpoint = ModelCheckpoint(filepath=r"C:\Users\Admin\Downloads\VGG16_NEW", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=4, verbose=1, mode='auto')

#H = model.fit_generator(steps_per_epoch= 1840 // batch_size,generator=traindata, validation_data= testdata, validation_steps=460// batch_size,epochs=EPOCHS,callbacks=[checkpoint, t_board, terminate_on_nan, reduce_lr])
#H = model.fit_generator(steps_per_epoch= 1840 // batch_size,generator=traindata, validation_data= testdata, validation_steps=460// batch_size,epochs=EPOCHS,callbacks=[checkpoint, early, terminate_on_nan, reduce_lr])

H = model.fit_generator(traindata,steps_per_epoch= int(1840/ batch_size),validation_data= testdata, validation_steps=int(460/ batch_size),epochs=EPOCHS,callbacks=[checkpoint,terminate_on_nan,reduce_lr])

