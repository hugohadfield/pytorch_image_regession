# Pytorch Image Regression
It is really common to find tutorials and examples of doing image **classification** but really hard to find simple examples of image **regression**, ie. predicting a vector from an image.

This repo is a super basic template for setting up an image regression task, loading images and csv data, training a network, and evaluating its performance on a test set.


# The simple regression task
The regression task set up here is purposefully simple. Our neural network should be able to absolutely crush this.

Given an image like this:

![example input image](example_dataset/train/images/image_0.png)

The network has to learn to regress a target which is the direction of the arrow as a 2d vector of length 1.0.
For this image the target answer is `0.9633736512703545,-0.2681626522057565`.


# The data loader
The main useful bit of this repo is probably `data_loading.py`. In this file we define how the image and target loading is done. Specifically we use a subclass of `torchvision.datasets.ImageFolder` but we overload attributes such that it sets everything up as a regression task rather than a classification one.


# The dataset
The dataset is in the folder `example_dataset`. It is split into 1000 training images and 100 test images.
For each of the `train` and `test` subdirectories there is also a csv file `train.csv` and `test.csv` respectively
that map the input image to the correct output answer.

This is what the first couple of lines of `train.csv` look like:
```
image_path,x,y
example_dataset/train/images/image_0.png,0.9633736512703545,-0.2681626522057565
example_dataset/train/images/image_1.png,0.39134577679470234,0.9202437084734407
```

The dataset is generated with the script `generate_sample_dataset.py`. If you just want to play around with the data or use this repo as a template for your own image regression task you won't need to run this script. If you do want to mess about with running it then you will also need to have `pandas` and OpenCV installed on your system.
