# Traffic Signs Classification
Classifying the [German Traffic Signs Recognition Benchmark dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) using a convolutional deep neural network in Keras, achieving an accuracy of <b><i>98.79%</b></i>, compared to human accuracy of 98.84% (from <i>[Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition](https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf)</i>).

## Requirements

* H5PY
* Keras
* NumPy
* Scikit-Image
* The German Traffic Signs Recognition Benchmark dataset.  
    Download:
    * <code>GT-final_test.csv</code> from https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip.
    * <code>Final_Training/</code> from https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
    * <code>Final_Test/</code> from https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip  

    Placing all downloaded files and directories in the <code>GTSRB/</code> folder in the project directory.


## Preprocessing

The pictures from the dataset vary in illumination, so the images are preprocessed using histogram equalization to yellow in HSV and the RGB axis is rolled to 0. The images also vary in size, so they are resized to be 48 x 48 x 3 and cropped to the center. The are also converted to numpy arrays, associated with labels, and one-hot encoded.

The is some variance in the number of samples between classes, so the images are augmented to increase the size of the dataset. This is done by...

1. Translating,
2. Rotating,
3. Shearing, and
4. Zooming in and out of the images.

## Model

The neural network uses a sequential model and has six convolutional layers, four dropout layers for regularization and preventing overfitting, and a flattened fully connected hidden layer.

Two features are implemented in training the model:

* A decaying learning rate.
* Model checkpoint - saving the model to prevent overfitting after too many epochs.

