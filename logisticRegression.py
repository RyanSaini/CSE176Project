# Import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat

def main():

    # Read data from MNISTmini.mat
    mnist_mini = loadmat("MNISTmini.mat")
    mnist_mini_data = mnist_mini["train_fea1"]
    mnist_mini_label = mnist_mini["train_gnd1"]
    mnist_mini_label = mnist_mini_label.flatten()

    # Extract all images of 6s and 8s
    images_of_6 = mnist_mini_data[mnist_mini_label == 6]
    images_of_8 = mnist_mini_data[mnist_mini_label == 8]

    # Select first 500 6s and 8s for training set
    training_6 = images_of_6[0:500]
    training_8 = images_of_8[0:500]

    # Select next 500 6s and 8s for validation set
    validation_6 = images_of_6[500: 1000]
    validation_8 = images_of_8[500: 1000]

    # Combine data in order to create training and validation sets
    training_X = np.concatenate((training_6, training_8))
    validation_X = np.concatenate((validation_6, validation_8))

    # Creates labels for the training data
    training_Y_6 = [6] * 500
    training_Y_8 = [8] * 500
    labels_Y = np.concatenate((training_Y_6, training_Y_8))

    training_X = normalizeData(training_X)

    # Making mean and std matrices
    #mean = np.mean(training_X, axis = 0)
    #std = np.std(training_X, axis = 0)

    # Data with 0 mean and data with 0 mean and unit variance
    #training_X_0_mean = training_X - mean
    #training_X_normalized = normalizeData(training_X)

    # Display mean image
    #image = mean
    #image = np.array(image, dtype='float')
    #pixels = image.reshape((10, 10))
    #plt.imshow(pixels, cmap='gray')
    #plt.show()

    # Display std image
    #image = std
    #image = np.array(image, dtype='float')
    #pixels = image.reshape((10, 10))
    #plt.imshow(pixels, cmap='gray')
    #plt.show()

    # Displays first image
    #image = training_X[0]
    #image = np.array(image, dtype='float')
    #pixels = image.reshape((10, 10))
    #plt.imshow(pixels, cmap='gray')
    #plt.show()

    # Displays first image with mean subtracted
    #first_image = training_X_0_mean[0]
    #first_image = np.array(first_image, dtype='float')
    #pixels = first_image.reshape((10, 10))
    #plt.imshow(pixels, cmap='gray')
    #plt.show()

    # Displays first image with mean subtracted and unit variance
    #first_image = training_X_normalized[0]
    #first_image = np.array(first_image, dtype='float')
    #pixels = first_image.reshape((10, 10))
    #plt.imshow(pixels, cmap='gray')
    #plt.show()

    # Create logistic regression models
    logReg = LogisticRegression(penalty = "l1", solver = "liblinear")

    # Trains model
    logReg.fit(training_X, labels_Y)

     # Prints out accuracy of model on validation set
    print(logReg.score(validation_X, labels_Y))


# Normalizes image data by subtracting mean and dividing by standard deviation
def normalizeData(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    std[std == 0] = 1

    data_normalized = (data - mean) / std

    return data_normalized

if __name__ == "__main__":
    main()