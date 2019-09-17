# Traffic Signs Classification
Classifying the [German Traffic Signs Recognition Benchmark dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) using a convolutional deep neural network, achieving an accuracy of <b><i>98.79%</b></i>, compared to human accuracy of 98.84% (from <i>[Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition](https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf)</i>).

## Preprocessing

The pictures from the dataset vary in illumination, so the images are preprocessed using histogram equalization to yellow in HSV. The images also vary in size, so they are resized to be 48 x 48 x 3. Cropping to the center, and rolling the RGB axis to 0 is performed.

The is some variance in the number of samples between classes, so the images are augmented to increase the size of the dataset. This is done by...

1. Translating,
2. Rotating,
3. Shearing, and
4. Zooming in and out of the images.
