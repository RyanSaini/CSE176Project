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

    # Select next 500 6s and 8s for test set
    test_6 = images_of_6[1000:1500]
    test_8 = images_of_8[1000:1500]

    # Combine data in order to create training, validation, and test sets
    training_X = np.concatenate((training_6, training_8))
    validation_X = np.concatenate((validation_6, validation_8))
    test_X = np.concatenate((test_6, test_8))

    # Creates labels for the training data
    training_Y_6 = [6] * 500
    training_Y_8 = [8] * 500
    labels_Y = np.concatenate((training_Y_6, training_Y_8))

    # Normalize data sets
    training_X = normalizeData(training_X)
    validation_X = normalizeData(validation_X)
    test_X = normalizeData(test_X)

    df_L1 = pd.DataFrame([], columns = ["C", "error"])
    df_L2 = pd.DataFrame([], columns = ["C", "error"])

    c = 0.01
    while c <= 2:
        # Create models with varying C values and with different penalties
        logReg_L1 = LogisticRegression(C = c, penalty = "l1", solver = "liblinear")
        logReg_L2 = LogisticRegression(C = c, penalty = "l2", solver = "liblinear")

        # Train different models
        logReg_L1.fit(training_X, labels_Y)
        logReg_L2.fit(training_X, labels_Y)

        # Create different rows of data for each model
        row_L1 = {"C" : c, "error" : 1 - logReg_L1.score(validation_X, labels_Y)}
        row_L2 = {"C" : c, "error" : 1 - logReg_L2.score(validation_X, labels_Y)}

        # Add each respective row to its dataframe
        df_L1 = df_L1.append(row_L1, ignore_index = True)
        df_L2 = df_L2.append(row_L2, ignore_index = True)

        c = c + 0.01

    ax = df_L1.plot(x = "C", y = "error", kind = "line", color = "red", label = "Model Using L1 penalty")
    df_L2.plot(x = "C", y = "error", kind = "line", ax = ax, color = "blue", label = "Model Using L2 penalty", title = "Validation Error with Varying C Values", ylabel = "Error")
    plt.legend()
    plt.show()

    # Print out model's score on the testing set
    logReg = LogisticRegression(C = 0.2)
    logReg.fit(training_X, labels_Y)
    print(logReg.score(test_X, labels_Y))

# Normalizes image data by subtracting mean and dividing by standard deviation
def normalizeData(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    std[std == 0] = 1

    data_normalized = (data - mean) / std

    return data_normalized

if __name__ == "__main__":
    main()