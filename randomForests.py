import time
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



    # Create random forest classifier model(n_estimator)
    # df = pd.DataFrame([], columns = ["num_trees", "error"])
    # num_trees = 1
    # while num_trees <= 100:
    #     randForest = RandomForestClassifier(n_estimators = num_trees)

    #     # Train model with training data
    #     randForest.fit(training_X, labels_Y)

    #     row = {"num_trees" : num_trees, "error" : 1 - randForest.score(validation_X, labels_Y)}

    #     df = df.append(row, ignore_index = True)

    #     num_trees = num_trees + 1

    # df.plot(x = "num_trees", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying n_estimators Values", ylabel = "Error", xlabel = "Number of Trees")
    # plt.show()
 
    # Create random forest classifier model(Max_Depth)
    # df = pd.DataFrame([], columns = ["max_depth", "error"])

    # num_trees = 100
    # max_depth = 1
    # while max_depth <= 100:
    #     randForest = RandomForestClassifier(max_depth = max_depth)

    #     # Train model with training data
    #     randForest.fit(training_X, labels_Y)

    #     row = {"max_depth" : max_depth, "error" : 1 - randForest.score(validation_X, labels_Y)}

    #     df = df.append(row, ignore_index = True)

    #     max_depth = max_depth + 1

    # df.plot(x = "max_depth", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying Max Depth Values", ylabel = "Error", xlabel = "Number of Trees")
    # plt.show()
    
    # Create random forest classifier model(min_samples_leaf)
    # df = pd.DataFrame([], columns = ["min_samples_leaf", "error"])

    # num_trees = 100
    # min_samples_leaf = 1
    # while min_samples_leaf <= 50:
    #     randForest = RandomForestClassifier(min_samples_leaf = min_samples_leaf)

    #     # Train model with training data
    #     randForest.fit(training_X, labels_Y)

    #     row = {"min_samples_leaf" : min_samples_leaf, "error" : 1 - randForest.score(validation_X, labels_Y)}

    #     df = df.append(row, ignore_index = True)

    #     min_samples_leaf = min_samples_leaf + 1

    # df.plot(x = "min_samples_leaf", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying min_samples_leaf Values", ylabel = "Error", xlabel = "Number of Trees")
    # plt.show()


    # Create random forest classifier model(random_state)
    df = pd.DataFrame([], columns = ["random_state", "error"])

    num_trees = 100
    random_state = 1
    while random_state <= 50:
        randForest = RandomForestClassifier(random_state = random_state)

        # Train model with training data
        randForest.fit(training_X, labels_Y)

        row = {"random_state" : random_state, "error" : 1 - randForest.score(validation_X, labels_Y)}

        df = df.append(row, ignore_index = True)

        random_state = random_state + 1

    df.plot(x = "random_state", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying random_state Values", ylabel = "Error", xlabel = "Number of Trees")
    plt.show()
    

    # Create random forest classifier model(random_state)
    df = pd.DataFrame([], columns = ["class_weight", "error"])

    num_trees = 100
    verbose = 0
    while verbose <= 50:
        randForest = RandomForestClassifier(verbose = verbose)

        # Train model with training data
        randForest.fit(training_X, labels_Y)

        row = {"verbose" : verbose, "error" : 1 - randForest.score(validation_X, labels_Y)}

        df = df.append(row, ignore_index = True)

        verbose = verbose + 1


    df.plot(x = "verbose", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying verbose Values", ylabel = "Error", xlabel = "Number of Trees")
    plt.show()


    # df4 = pd.DataFrame([], columns = ["n_jobs", "error"])
    # df4_time = pd.DataFrame([], columns = ["n_jobs", "time"])
    # n_jobs = 1
    # while n_jobs <= 100:
    #     start_time = time.time()
    #     randForest = RandomForestClassifier(n_estimators = 100, n_jobs = n_jobs)

    #     randForest.fit(training_X, labels_Y)

    #     row = {"n_jobs" : n_jobs, "error" : 1 - randForest.score(validation_X, labels_Y)}

    #     row_time = {"n_jobs" : n_jobs, "time" : time.time() - start_time}

    #     df4 = df4.append(row, ignore_index = True)
    #     df4_time = df4_time.append(row_time, ignore_index = True)

    #     n_jobs = n_jobs + 1

    # df4.plot(x = "n_jobs", y = "error", kind = "line", color = "blue", title = "Validation Error with Varying n_jobs Values", ylabel = "Error", xlabel = "n_jobs")
    # plt.show()
    # df4_time.plot(x = "n_jobs", y = "time", kind = "line", color = "red", title = "Time to Train and Run Validation Set on Classifier with Varying n_jobs Values", ylabel = "Time", xlabel = "n_jobs")
    # plt.show()




def normalizeData(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    std[std == 0] = 1

    data_normalized = (data - mean) / std

    return data_normalized

if __name__ == "__main__":
    main()