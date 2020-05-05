from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, abspath


#The program uses the train and val directory in the getcards directory to train and validate the model


def train():

    print("setting up the model")
    batch_size = 5
    IMG_HEIGHT= 680
    IMG_WIDTH= 488
    epochs = 400

    PATH = os.path.join(os.path.dirname("getCards/Cards/"))
    train_dir = os.path.join(PATH, "train")
    validation_dir = os.path.join(PATH,"val")


    #image generation to increase the data size of the train data set
    image_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=90,
        width_shift_range=.15,
        height_shift_range=.15,
        vertical_flip=True,
        zoom_range=0.258
                        )

    #setting up the data generation for training and validation data generation
    train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                               class_mode='categorical'
                                               )
    #applying rescaling to the images in order to not let the quality of the image affect the training of the model
    image_gen_val = ImageDataGenerator(rescale=1./255)
    val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                     directory=validation_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')

    #adding different layers to the model, compiling it, training, and then saving it
    model = Sequential()
    model.add(Dense(60, activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(240, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(480, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(train_data_gen.num_classes, activation='softmax'))

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    training_steps = train_data_gen.n // train_data_gen.batch_size
    val_steps = val_data_gen.n // val_data_gen.batch_size
    print("training model")
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=  training_steps,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps= val_steps
    )

    print("saving model")
    model.save("MagicCardIR.h5")

