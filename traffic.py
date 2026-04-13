import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    images, labels = load_data(sys.argv[1])

    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()

    model.fit(x_train, y_train, epochs=EPOCHS)

    model.evaluate(x_test, y_test, verbose=2)

    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))

        if not os.path.isdir(category_path):
            print(f"[WARNING] Category folder not found: {category_path}")
            continue

        for filename in os.listdir(category_path):
            filepath = os.path.join(category_path, filename)

            img = cv2.imread(filepath)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            img = img / 255.0

            images.append(img)
            labels.append(category)

    print(f"[INFO] Loaded {len(images)} images across {NUM_CATEGORIES} categories.")
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.

    Architecture:
    - Two Conv2D + MaxPooling blocks for feature extraction
    - Flatten + Dense layers for classification
    - Dropout to reduce overfitting
    - Softmax output for multi-class classification
    """
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),

        
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


if __name__ == "__main__":
    main()