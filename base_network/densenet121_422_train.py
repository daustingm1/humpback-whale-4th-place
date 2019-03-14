import os
import ast
import datetime as dt
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import math
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Input, Reshape, GlobalAveragePooling2D
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential, Model, load_model, save_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger, LearningRateScheduler
from keras.optimizers import Adam, SGD
from keras.applications import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.utils import multi_gpu_model
from keras.utils import to_categorical, Sequence
import time
from tqdm import tqdm

start_time = time.time()

num_classes=422

df_train = pd.read_csv('../input/initial_train_422_classes.csv')
df_val = pd.read_csv('../input/initial_val_422_classes.csv')

train_imgs = df_train['Image']
val_imgs = df_val['Image']
train_labels = to_categorical(df_train['label'], num_classes=num_classes)
val_labels = to_categorical(df_val['label'], num_classes=num_classes)

input_size = 256
batch_size=32
epochs=100

def mixup(x, y, alpha=0.3, u=0.5):
    if np.random.random() < u:
        lam = np.random.beta(alpha, alpha)
        batsize = len(x)
        index=np.random.permutation(batsize)
        x = lam * x + (1-lam) * x[index, :]
        y = lam * y + (1 - lam) * y[index, :]
    return x, y

def train_generator():
    while True:
        for start in range(0, len(train_imgs), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(train_imgs))
            ids_train_batch = train_imgs[start:end]
            y_batch = train_labels[start:end]
            for id in ids_train_batch.values:
                img = cv2.imread('../input/train/{}'.format(id), 0)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img, (input_size, input_size))
                x_batch.append(img)
            x_batch = np.array(x_batch) #, np.float32) / 255
            x_batch = preprocess_input(x_batch)
            #y_batch = np.array(y_batch, np.float32) / 255
            #x_batch, y_batch = mixup(x=x_batch, y=y_batch)
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, len(val_imgs), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(val_imgs))
            ids_train_batch = val_imgs[start:end]
            y_batch = val_labels[start:end]
            for id in ids_train_batch.values:
                img = cv2.imread('../input/train/{}'.format(id),0)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img, (input_size, input_size))
                x_batch.append(img)
            x_batch = np.array(x_batch) #, np.float32) / 255
            x_batch = preprocess_input(x_batch)
            yield x_batch, y_batch

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def step_decay(epoch):
    initial_lrate = 0.0002
    drop=0.5
    epochs_drop = 5.0
    lrate = initial_lrate*math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=256, n_classes=200, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_labels_temp = [self.labels[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_labels_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        y = np.empty((self.batch_size), dtype=int)
        x_batch = []
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            img = cv2.imread('../class_problem_only/all_traintest2/{}'.format(ID), 0)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (self.dim, self.dim))
            x_batch.append(img)

            # Store class
            y[i] = list_labels_temp[i]
        x_batch = np.array(x_batch) #, np.float32) / 255
        X = preprocess_input(x_batch)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

base_model = DenseNet121(input_shape=(input_size, input_size, 3), weights=None, include_top=False)
base_model.load_weights('weights/densenet121_200_classes.15-0.23897_head_weights_only.hdf5')  #load from previous best model on top 200 classes
for (i, layer) in enumerate(base_model.layers[:]):
    layer.trainable = False

x = GlobalAveragePooling2D(name='global_average_pooling_x')(base_model.output)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation='softmax', name='dense_x')(x)
model = Model(inputs=[base_model.input], outputs=[output])

print(model.summary())

model.compile(optimizer=Adam(lr=0.0002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

lrate=LearningRateScheduler(step_decay)

callbacks = [lrate, EarlyStopping(monitor='val_loss',
                           patience=12,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=8,
                               verbose=1,
                               min_delta=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/densenet121_422_classes_head_fine_tune.{epoch:02d}-{val_loss:.5f}.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs'),
             CSVLogger('logs/densenet121_422_classes_head_fine_tune{}.csv'.format(start_time), separator=',')]

training_generator = DataGenerator(train_imgs, df_train['label'], n_classes=num_classes)
validation_generator = DataGenerator(val_imgs, df_val['label'], n_classes=num_classes)

model.fit_generator(generator=training_generator,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks, initial_epoch=0,
                    validation_data=validation_generator, use_multiprocessing=True,
                    workers=8, max_queue_size=12)

model.load_weights('weights/densenet121_422_classes_head_fine_tune.40-0.93280.hdf5')

for (i, layer) in enumerate(model.layers[:]):
    layer.trainable = True

model.compile(optimizer=Adam(lr=0.0002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

lrate=LearningRateScheduler(step_decay)

callbacks = [lrate, EarlyStopping(monitor='val_loss',
                           patience=12,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=8,
                               verbose=1,
                               min_delta=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/densenet_422_classes_head_all_layers.{epoch:02d}-{val_loss:.5f}.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs'),
             CSVLogger('logs/densenet_epochlog_5422_classes_head_all_layers{}.csv'.format(start_time), separator=',')]

model.fit_generator(generator=training_generator,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks, initial_epoch=0,
                    validation_data=validation_generator, use_multiprocessing=True,
                    workers=8, max_queue_size=12)

model.load_weights('weights/densenet121_422_classes_head_all_layers.15-0.22215.hdf5')

model2 = Model(inputs=model.input, outputs=model.layers[-4].output)

model2.save_weights('weights/densenet121_422_classes_head_all_layers.15-0.22215_base_weights_only.hdf5')
