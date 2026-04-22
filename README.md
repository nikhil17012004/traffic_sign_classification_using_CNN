Traffic Sign Classification using CNN

A Convolutional Neural Network (CNN) built with TensorFlow and OpenCV to classify road signs into 43 categories using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. Achieves ~95%+ accuracy on the test set after 10 epochs.

What It Does:
Loads and preprocesses thousands of road sign images using OpenCV
Trains a CNN to classify signs into 43 categories (stop signs, speed limits, yield signs, etc.)
Evaluates model accuracy on a held-out test set
Optionally saves the trained model to a .h5 file for reuse

Approach:
- Pixel Normalization (img / 255.0)
- Two-Stage Conv2D + MaxPooling Feature Extraction
    Conv2D(32 filters, 3×3)  →  MaxPooling2D(2×2)
    Conv2D(64 filters, 3×3)  →  MaxPooling2D(2×2)
- ReLU Activation in Conv Layers
- One-Hot Encoding of Labels
    labels = tf.keras.utils.to_categorical(labels) 
    [ Label 2  →  [0, 0, 1, 0, 0, ... 0]   (43 elements, 1 at index 2) ]

How It Works:
load_data() - Walks through each of the 43 category folders, reads images with OpenCV, resizes them to 30×30, normalizes pixel values to [0, 1], and returns a list of image arrays and their corresponding labels.
get_model() - Builds a Sequential CNN with two convolutional + pooling blocks for feature extraction, followed by fully connected layers with dropout for regularization, and a softmax output layer for multi-class classification.

How to execute:

- Download the dataset using this link:
"https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip"
  
- Install dependencies:
pip install tensorflow opencv-python scikit-learn numpy

- Train the model:
python traffic.py gtsrb/

- Train and save the model:
python traffic.py gtsrb/ model.h5
