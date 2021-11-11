import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand

from sklearn.ensemble import RandomForestClassifier
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

    # Normalize data
    training_X = normalizeData(training_X)
    validation_X = normalizeData(validation_X)
    test_X = normalizeData(test_X)

    # Creates labels for the training data
    training_Y_6 = [6] * 500
    training_Y_8 = [8] * 500
    labels_Y = np.concatenate((training_Y_6, training_Y_8))

    df = pd.DataFrame([], columns = ["num_trees", "error"])
    # Create random forest classifier model
    num_trees = 1
    while num_trees <= 1000:
        randForest = RandomForestClassifier(n_estimators = num_trees)

        # Train model with training data
        randForest.fit(training_X, labels_Y)

        row = {"num_trees" : num_trees, "error" : 1 - randForest.score(validation_X, labels_Y)}

        df = df.append(row, ignore_index = True)

        num_trees = num_trees + 1

    df.plot(x = "num_trees", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying n_estimators Values", ylabel = "Error", xlabel = "Number of Trees")
    plt.show()

def normalizeData(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    std[std == 0] = 1

    data_normalized = (data - mean) / std

    return data_normalized

if __name__ == "__main__":
    main()