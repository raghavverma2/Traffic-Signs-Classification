import os
import sys
from glob import glob
import h5py
import time

import numpy as np

from skimage import io, color, exposure, transform
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend
backend.set_image_data_format('channels_first')

from matplotlib import pyplot

NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess(image):
    
    # Histogram normalization in HSV yellow
    temp = color.rgb2hsv(image)
    temp[:,:,2] = exposure.equalize_hist(temp[:,:,2])
    image = color.hsv2rgb(temp)
    
    # Crop central region
    ms = min(image.shape[:-1])
    center = image.shape[0]//2, image.shape[1]//2
    image = image[
        center[0] - ms//2 : center[0] + ms//2,
        center[1] - ms//2 : center[1] + ms//2,
        :
    ]
    
    # Resize
    image = transform.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Roll RGB axis to 0
    image = np.rollaxis(image, -1)
    
    return image

try:
    X = h5py.File('data.h5')['images'][:]
    Y = h5py.File('data.h5')['labels'][:]
    print("Using preprocessed images from data.h5")
except (IOError, OSError, KeyError):
    print("Could not find preprocessed data ['data.h5']\n")
    root = 'GTSRB/Final_Training/Images/'
    images = []
    labels = []
    paths = glob(os.path.join(root, '*/*.ppm'))
    np.random.shuffle(paths)
    total = len(paths)

    start = time.time()
    print("\nTotal training images:\t{}".format(total))
    for i in range(total):
        sys.stdout.write('\r')
        sys.stdout.write("Processed image # \t{} | {}% complete".format(i, round(i*100/total, 2)))
        sys.stdout.flush()

        path = paths[i]
        image = preprocess(io.imread(path))
        label = int(path.split('\\')[-2])
        images.append(image)
        labels.append(label)
    end = time.time()
    print("\nFinished preprocessing!\n")

    X = np.array(images, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
    h5py.File('data.h5').create_dataset('images', data = X)
    h5py.File('data.h5').create_dataset('labels', data = Y)
    print("Saved preprocessed data in data.h5")
    
    print("\nTime spent preprocessing: {} seconds".format(round(end - start, 2)))

X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                  test_size = 0.2,
                                                  random_state = 100)

datagen = ImageDataGenerator(featurewise_center = False,
                            featurewise_std_normalization = False,
                            width_shift_range = 0.1,
                            height_shift_range = 0.1,
                            shear_range = 0.1,
                            zoom_range = 0.2,
                            rotation_range = 10.0)
datagen.fit(X_train)

model = Sequential()

model.add(Conv2D(32,
                (3, 3),
                padding='same',
                input_shape = (3, IMG_SIZE, IMG_SIZE),
                activation = 'relu'))
model.add(Conv2D(32,
                (3, 3),
                activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,
                (3, 3),
                padding='same',
                activation = 'relu'))
model.add(Conv2D(64, (3, 3),
                activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128,
                (3, 3),
                padding='same',
                activation = 'relu'))
model.add(Conv2D(128, (3, 3),
                activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512,
               activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES,
               activation = 'relu'))

learning_rate = 0.01

model.compile(loss = 'categorical_crossentropy',
             optimizer = SGD(lr = learning_rate,
                            decay = 1e-6,
                            momentum = 0.9,
                            nesterov = True),
              metrics = ['accuracy'])

epochs = 30
bs = 32

model.fit_generator(datagen.flow(X_train, Y_train, batch_size = bs),
                   steps_per_epoch = X_train.shape[0],
                   epochs = epochs,
                   validation_data = (X_val, Y_val))