import numpy as np
import cv2
import glob
from sklearn import svm


############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/**
# use 15x15 keypoints on a uniform grid for each image with keypoint size 11 (like in Assignment 2.4 image retrieval)

# 2. Build the X_train matrix and y_train vector
# The k-th row in X_train (X_train[k]) contains the description of a training image which has the label y_train[k]
# This results in a shape for X_train of (num_train_images, num_keypoints*num_entry_per_keypoint)
# and a shape for y_train of (num_train_images),
# where:
# num_train_images: Number of train images
# num_keypoints: Number of keypoints per image (in our case 15*15)
# num_entry_per_keypoint: size of one feature descriptor (in our case 128)

# At the moment each descriptor (set of feature descriptors) for an image is a matrix,
# therefore it needs to be flattened in one vector.
# This means you must convert the matrix in a vector (you can use ndarray.flatten)

# The y_train vector contains the labels. Usually they are encoded as integers,
# where e.g.: 0 = flower, 1 = car and 2 = face,
# but you can use strings like "flower", "car" and "face" as well

# 3. We can now use scikit-learn to train a SVM classifier.
# Specifically we use a LinearSVC (Support Vector Classifier) in our case. Have a look at the documentation,
# you will need .fit(X_train, y_train)

# 4. We test on a variety of test images ./images/db/test/* by extracting an image descriptor in vector form.
# Its the same way we did for the training (except for a single image now)
# You can use .predict(list_of_descriptors_to_be_predicted) (check LinearSVC documentation again)
# to classify the image

# 5. Output the predicted classes.
