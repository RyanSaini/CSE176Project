import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand
import time

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


    # Plots validation error with varying number of trees
    df = pd.DataFrame([], columns = ["num_trees", "error"])
    num_trees = 1
    while num_trees <= 100:
        randForest = RandomForestClassifier(n_estimators = num_trees)

        # Train model with training data
        randForest.fit(training_X, labels_Y)

        row = {"num_trees" : num_trees, "error" : 1 - randForest.score(validation_X, labels_Y)}

        df = df.append(row, ignore_index = True)

        num_trees = num_trees + 1

    df.plot(x = "num_trees", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying n_estimators Values", ylabel = "Error", xlabel = "Number of Trees")
    plt.show()


    # Plots validation error with varying number of minimum samples needed to split as an int value
    df2 = pd.DataFrame([], columns = ["split", "error"])
    split = 2
    while split < 100:
        randForest = RandomForestClassifier(n_estimators = 10, min_samples_split = split)

        randForest.fit(training_X, labels_Y)

        row = {"split" : split, "error" : 1 - randForest.score(validation_X, labels_Y)}

        df2 = df2.append(row, ignore_index = True)

        split = split + 1

    df2.plot(x = "split", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying min_samples_split Values As Int", ylabel = "Error", xlabel = "Minimum Number of Samples to Split")
    plt.show()


    # Plots validation error with varying number of minimum samples needed to split as a float value
    df3 = pd.DataFrame([], columns = ["split", "error"])
    split = 0.1
    while split <= 1:
        randForest = RandomForestClassifier(n_estimators = 10, min_samples_split = split)

        randForest.fit(training_X, labels_Y)

        row = {"split" : split, "error" : 1 - randForest.score(validation_X, labels_Y)}

        df3 = df3.append(row, ignore_index = True)

        split = split + 0.1

    df3.plot(x = "split", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying min_samples_split Values As Float", ylabel = "Error", xlabel = "Minimum Number of Samples to Split")
    plt.show()


    # Plots validation error with varying number of n_jobs
    df4 = pd.DataFrame([], columns = ["n_jobs", "error"])
    df4_time = pd.DataFrame([], columns = ["n_jobs", "time"])
    n_jobs = -16
    while n_jobs <= 16:

        if n_jobs == 0:
            n_jobs = n_jobs + 1

        start_time = time.time()
        randForest = RandomForestClassifier(n_estimators = 100, n_jobs = n_jobs)

        randForest.fit(training_X, labels_Y)

        row = {"n_jobs" : n_jobs, "error" : 1 - randForest.score(validation_X, labels_Y)}
        row_time = {"n_jobs" : n_jobs, "time" : time.time() - start_time}

        df4 = df4.append(row, ignore_index = True)
        df4_time = df4_time.append(row_time, ignore_index = True)

        n_jobs = n_jobs + 1

    df4.plot(x = "n_jobs", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying n_jobs Values", ylabel = "Error", xlabel = "n_jobs")
    plt.show()
    df4_time.plot(x = "n_jobs", y = "time", kind = "line", color = "red", title = "Time to Train and Run Validation Set on Classifier with Varying n_jobs Values", ylabel = "Time", xlabel = "n_jobs")
    plt.show()

    # Plots validation error with varying ccp_alpha values
    df5 = pd.DataFrame([], columns = ["ccp_alpha", "error"])
    ccp_alpha = 0.001
    while ccp_alpha <= 0.2:

        randForest = RandomForestClassifier(ccp_alpha = ccp_alpha)

        randForest.fit(training_X, labels_Y)

        row = {"ccp_alpha" : ccp_alpha, "error" : 1 - randForest.score(validation_X, labels_Y)}

        df5 = df5.append(row, ignore_index = True)

        ccp_alpha = ccp_alpha + 0.001

    df5.plot(x = "ccp_alpha", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying ccp_alpha Values", ylabel = "Error", xlabel = "ccp_alpha")
    plt.show()


    # Plots validation error with varying max_samples values
    df6 = pd.DataFrame([], columns = ["max_samples", "error"])
    max_samples = 1
    while max_samples <= 100:

        randForest = RandomForestClassifier(max_samples = max_samples)

        randForest.fit(training_X, labels_Y)

        row = {"max_samples" : max_samples, "error" : 1 - randForest.score(validation_X, labels_Y)}

        df6 = df6.append(row, ignore_index = True)

        max_samples = max_samples + 1

    df6.plot(x = "max_samples", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying max_samples Values", ylabel = "Error", xlabel = "max_samples")
    plt.show()


# Funcation to normalize data
def normalizeData(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    std[std == 0] = 1

    data_normalized = (data - mean) / std

    return data_normalized

if __name__ == "__main__":
    main()