# Import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat

def main():

    # Read data from MNISTmini.mat
    mnist_mini = loadmat("data/MNISTmini.mat")
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
    training_X = normalizeData(training_X)
    validation_X = np.concatenate((validation_6, validation_8))

    # Creates labels for the training data
    training_Y_6 = [6] * 500
    training_Y_8 = [8] * 500
    labels_Y = np.concatenate((training_Y_6, training_Y_8))

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