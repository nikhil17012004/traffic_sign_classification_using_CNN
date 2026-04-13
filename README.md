Traffic Sign Classification using CNN

A Convolutional Neural Network (CNN) built with TensorFlow and OpenCV to classify road signs into 43 categories using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. Achieves ~95%+ accuracy on the test set after 10 epochs.

What It Does:
Loads and preprocesses thousands of road sign images using OpenCV
Trains a CNN to classify signs into 43 categories (stop signs, speed limits, yield signs, etc.)
Evaluates model accuracy on a held-out test set
Optionally saves the trained model to a .h5 file for reuse

How It Works:
load_data() — Walks through each of the 43 category folders, reads images with OpenCV, resizes them to 30×30, normalizes pixel values to [0, 1], and returns a list of image arrays and their corresponding labels.
get_model() — Builds a Sequential CNN with two convolutional + pooling blocks for feature extraction, followed by fully connected layers with dropout for regularization, and a softmax output layer for multi-class classification.
