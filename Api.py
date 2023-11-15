import cv2 as cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras import optimizers
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


class Api:
    def read_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # img = cv2.medianBlur(img, 5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def save_img(self, img, fileName):
        plt.imshow(img)
        # print(img, fileName);
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join('static/', fileName), rgbImg)
        cv2.waitKey(0)
        return 'save image' + fileName + 'successful'

    def load_image(self, path, show=False):
        img = image.load_img(path, target_size=(300, 300))
        img = (np.asarray(img))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        return img_tensor

    def loadModel(self, path, img_tensor):
        model_loaded = load_model(path, compile=False)
        rmsprop = optimizers.RMSprop(
            lr=0.0001, rho=0.9, epsilon=1e-08)
        model_loaded.compile(loss='categorical_crossentropy',
                             optimizer=rmsprop, metrics=['accuracy'])
        model_loaded.summary()
        print('Load Model Successfull')

        return model_loaded

    def predict(self, model, img_tensor):
        predict = model.predict(img_tensor)
        print('value sermentation', np.argmax(
            model.predict(img_tensor), axis=-1))
        print('result', predict[0])

        result = None
        if predict[0][0] == 1:
            result = 'Diffuse'
        elif predict[0][1] == 1:
            result = 'Intestinal'
        else:
            result = 'Tidak terdeteksi'

        return result
