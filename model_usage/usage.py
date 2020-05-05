from tensorflow.keras.models import load_model
import cv2
import os
from glob import glob
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def use():


    #loading the model, using the train directories to get a list of card names, and then use the model prediction on the image
    #NOTE be careful of the height and width of the input image
    FINAL_SIZE_H = 680
    FINAL_SIZE_W = 488
    model = load_model("MagicCardIR.h5")

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    PATH = os.path.join(os.path.dirname("model_usage/image/"))
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            PATH,
            target_size=(FINAL_SIZE_H, FINAL_SIZE_W),
            color_mode="rgb",
            shuffle = False,
            class_mode='categorical',
            batch_size=1)
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    prob = model.predict_generator(test_generator, steps=nb_samples)
    predicted_class_indices = np.argmax(prob, axis=1)
    print(predicted_class_indices)
    class_names = glob("getCards/cards/train/*")
    class_names = sorted(class_names)
    name_id_map = dict(zip(class_names, range(len(class_names))))
#
    for name, value in name_id_map.items():
        if value == predicted_class_indices:
            print(name)
