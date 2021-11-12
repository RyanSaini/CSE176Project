# CSE176Project

This is for UC Merced Class CSE 176 Project 1.

The objective of this project is to do binary classification on two handwritten digits from the MNISTmini dataset. This project uses logistic regression and random
forests models in order to classify the two digits. Logistic regression models and random forests classifiers are made using the scikit-learn implementation. For the logistic regression model, the L1 and L2 penalties were tested with differing C values. For the random forest model, the number of trees in the forest, the minimum number of samples required to be at a leaf node, the random state of the tree, the number of jobs to run in parallel while building the tree, the max samples to draw from our dataset to train each base estimator, and the complexity parameter used for minimal cost-complexity pruning were altered.

Relevant Links:

1) https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
2) https://scikit-learn.org/stable/modules/linear_model.html
3) https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
4) https://scikit-learn.org/stable/modules/ensemble.html
