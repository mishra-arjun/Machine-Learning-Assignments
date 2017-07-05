# Importing packages

import pandas as pd
import numpy as np
import scipy as sc
from operator import itemgetter

# Reading in the dataset
sonar = pd.read_csv("sonar.csv")

# Looking at the labels in the dataset
print("Labels in the dataset: ", np.unique(sonar.iloc[:, [60]])) # Only M and R

# Converting labels to binary and making Y variable
Y = np.array([sonar.iloc[: , 60] == "M"]).flatten() * 1

# Converting dataframe to a matrix
X = np.array(sonar.iloc[: , :60])

# Also making the complete dataset
full_data = np.concatenate((X, np.reshape(Y, (207, 1))), axis = 1)


# Now we will start to build the tree

# The split of a tree is going to be decided based on the Gini index.
# A decision tree is implemented using the following algorithm:
# Pick a random feature - then pick a random value from that feature and then split the dataset on this value.
# Using Gini will ensure that the feature and value picked is not random.

# Defining Gini - with weights as a dummy argument here - will be used in adaboost
def gini_index_1(data, response, no_of_features = 30, no_of_feat_val = 5):
    # Picking out random features from the dataset
    rand_feat_index = np.random.choice(data.shape[1], no_of_features)
    gini_lv2_list = []
    for f_ind in rand_feat_index:
        # Picking out random values from the feature
        # If the number of data points is lesser than the no of values we choose,
        # choose the number of values = data length.
        rand_feat_value = np.random.choice(data[:, f_ind], min(no_of_feat_val, data.shape[0]))

        gini_lv1_list = []
        for f_val in rand_feat_value:
            # For simplicity, we are not including extreme points where all other values are either greater or smaller
            # To avoid overfitting, we will put a condition on the minimum number of values in a split
            if sum(data[:, f_ind] < f_val) == 0 or sum(data[:, f_ind] > f_val) == 0:
                continue
            small_prop_sum = (np.mean(response[data[:, f_ind] < f_val]) * (1 - np.mean(response[data[:, f_ind] < f_val])))**2
            large_prop_sum = (np.mean(response[data[:, f_ind] >= f_val]) * (1 - np.mean(response[data[:, f_ind] >= f_val])))** 2
            gini_lv1 = small_prop_sum + large_prop_sum
            gini_lv1_list.append((f_ind, f_val, gini_lv1))

        if len(gini_lv1_list) == 0:
            gini_lv1_list.append((f_ind, np.random.choice(rand_feat_value, 1), 0.9999999999))
        gini_lv1_best = min(gini_lv1_list, key = itemgetter(2))
        gini_lv2_list.append(gini_lv1_best)

    return min(gini_lv2_list, key = itemgetter(2))[0:2]
    # This will return the feature index and feature value, respectively


# Writing the function for the tree
# To stop splitting the tree, we will use the condition on minimum records in leaf

# Importing the defaultdict module
from collections import defaultdict

# Writing the function for creating the tree
def greedy_tree(data, no_of_features = 30, no_of_feat_val = 10, leaf_min_rec = 10, gini_index = gini_index_1):

    tree_store = defaultdict(defaultdict)

    # First we will split the data based on our gini function
    root_feat_index, root_feat_val = gini_index(data[:, :-1], data[:, -1], no_of_features, no_of_feat_val)

    left = data[data[:, root_feat_index] >= root_feat_val, : ]
    right = data[data[:, root_feat_index] < root_feat_val, : ]

    if left.shape[0] == 0 or right.shape[0] == 0:
        tree_store["node_parent_feat_ind"] = root_feat_index
        tree_store["node_parent_feat_val"] = root_feat_val
        tree_store["left_prediction"] = int(np.random.choice([0, 1], 1))
        tree_store["right_prediction"] = int(np.random.choice([0, 1], 1))
        tree_store["records in left leaf"] = left.shape[0]
        tree_store["records in right leaf"] = right.shape[0]

    elif left.shape[0] <= leaf_min_rec and right.shape[0] <= leaf_min_rec:
        pred_left = int(np.mean(left[: , left.shape[1] - 1]) > 0.5)

        pred_right = int(np.mean(right[: , right.shape[1] - 1]) > 0.5)

        tree_store["node_parent_feat_ind"] = root_feat_index
        tree_store["node_parent_feat_val"] = root_feat_val
        tree_store["left_prediction"] = pred_left
        tree_store["right_prediction"] = pred_right
        tree_store["records in left leaf"] = left.shape[0]
        tree_store["records in right leaf"] = right.shape[0]

    elif left.shape[0] <= leaf_min_rec and right.shape[0] > leaf_min_rec:
        pred_left = int(np.mean(left[:, left.shape[1] - 1]) > 0.5)
        tree_store["node_parent_feat_ind"] = root_feat_index
        tree_store["node_parent_feat_val"] = root_feat_val
        tree_store["left_prediction"] = pred_left
        tree_store["records in left leaf"] = left.shape[0]
        tree_store["sub_tree_right"] = greedy_tree(right, no_of_features, no_of_feat_val, leaf_min_rec)

    elif left.shape[0] > leaf_min_rec and right.shape[0] <= leaf_min_rec:
        pred_right = int(np.mean(right[:, right.shape[1] - 1]) > 0.5)
        tree_store["node_parent_feat_ind"] = root_feat_index
        tree_store["node_parent_feat_val"] = root_feat_val
        tree_store["right_prediction"] = pred_right
        tree_store["records in right leaf"] = right.shape[0]
        tree_store["sub_tree_left"] = greedy_tree(left, no_of_features, no_of_feat_val, leaf_min_rec)

    else:
        tree_store["node_parent_feat_ind"] = root_feat_index
        tree_store["node_parent_feat_val"] = root_feat_val
        tree_store["sub_tree_left"] = greedy_tree(left, no_of_features, no_of_feat_val, leaf_min_rec)
        tree_store["sub_tree_right"] = greedy_tree(right, no_of_features, no_of_feat_val, leaf_min_rec)

    return tree_store


# Making the tree model on the whole dataset
# For this lone tree, we will use all the features
tree_output = greedy_tree(full_data, no_of_features = 60, no_of_feat_val = 5, leaf_min_rec = 10)

# Defining function to make predictions on the data using the tree model
def tree_predict(tree_model, test_data_row):
    if test_data_row[tree_model["node_parent_feat_ind"]] >= tree_model["node_parent_feat_val"] and\
                    isinstance(tree_model["left_prediction"], int):
        return tree_model["left_prediction"]
    elif test_data_row[tree_model["node_parent_feat_ind"]] < tree_model["node_parent_feat_val"] and \
            isinstance(tree_model["right_prediction"], int):
        return tree_model["right_prediction"]
    elif test_data_row[tree_model["node_parent_feat_ind"]] >= tree_model["node_parent_feat_val"] and\
                    len(tree_model["left_prediction"]) == 0:
        return tree_predict(tree_model["sub_tree_left"], test_data_row)
    elif test_data_row[tree_model["node_parent_feat_ind"]] < tree_model["node_parent_feat_val"] and\
                    len(tree_model["right_prediction"]) == 0:
        return tree_predict(tree_model["sub_tree_right"], test_data_row)

# Making the predictions on the whole dataset
pred_list = []
for row in X:
    pred_list.append(tree_predict(tree_output, row))

# Converting list to an array
pred_list = np.array(pred_list)

# Computing the confusion matrix of the predictions on the whole dataset
from sklearn.metrics import confusion_matrix, accuracy_score
print("The confusion matrix for prediction on the whole dataset:\n", confusion_matrix(Y, pred_list))
print("The accuracy score for prediction on the whole dataset:\n", accuracy_score(Y, pred_list))

# For the single tree, we are getting 100% accuracy. This clearly tells us that our tree is overfitting.

# =============================================== Part B ===============================================================

# Implementing cross validation
from sklearn.model_selection import KFold
folds = KFold(n_splits=10, shuffle=True)

# Initializing accuracy storage lists
avg_acc = list()
fold_acc = list()

# Running 100 iterations on 10 cross validation trees - 1000 trees
for iter in range(100):

    for train, test in folds.split(X):
        train_data, test_data = full_data[train], full_data[test]

        tree_model = greedy_tree(train_data, no_of_features = 30, no_of_feat_val = 5, leaf_min_rec = 20)
        y_pred = []
        for row in test_data[:, :-1]:
            y_pred.append(tree_predict(tree_model, row))
        y_pred = np.array(y_pred)

        fold_acc.append(accuracy_score(test_data[:,-1], y_pred))

    avg_acc.append(np.mean(fold_acc))
    print("Done with iteration: {}. Average accuracy score for iteration: {}".format(iter, np.mean(fold_acc)))

# Calculating the mean and standard deviation of the predictions

cv_mean = np.mean(avg_acc)
cv_sd = np.std(avg_acc)

print("The mean of the accuracies is:", cv_mean)
print("The standard deviation of the accuracies is:", cv_sd)


# ================================================= Part C ============================================================

# As we saw in the above exercise, the prediction of the tree on the whole dataset was 100% but dropped to ~60% when
# the model was used on test data that it hadn't seen. This is a clear case of OVERFITTING.

# Here we will implement 2 changes in the Gini function:
# 1. Tree pruning: To avoid overfitting, I will put a condition on the minimum number of data points that can be
# present in a leaf. This way, a leaf will not have any lesser than "N" number of points in it. With this condition
# we ensure that the decision boundaries are general and not fitting each and every point separately.

# 2. Feature value selection: Instead of selecting values randomly for each feature, we will pick values
# evenly spaced between the minima and the maxima of each feature we select. This improvement should result
# in better generalization of the decision boundaries and thus better prediction. Also, this will reduce the need
# to select and fit a large number of values as it will cover the whole range of values of the features. This
# will result in potential improvement in performance as we will need to select less number of values.


def gini_index_2(data, response, no_of_features = 30, no_of_feat_val = 5, tree_prune_cutoff = 5):
    # Picking out random features from the dataset
    rand_feat_index = np.random.choice(data.shape[1], no_of_features)
    gini_lv2_list = []
    for f_ind in rand_feat_index:
        # Picking out evenly spaced values between the minimum and maximum of the feature
        # If the number of data points is lesser than the no of values we choose,
        # choose the number of values = data length.

        # CHANGE #1
        rand_feat_value = np.linspace(min(data[:, f_ind]), max(data[:, f_ind]),
                                      min(no_of_feat_val, data.shape[0]), endpoint=False)

        gini_lv1_list = []
        for f_val in rand_feat_value:
            # For simplicity, we are not including extreme points where all other values are either greater or smaller
            # To avoid overfitting, we will put a condition on the minimum number of values in a split

            # CHANGE #2
            if sum(data[:, f_ind] < f_val) < tree_prune_cutoff or sum(data[:, f_ind] > f_val) < tree_prune_cutoff:
                continue
            small_prop_sum = (np.mean(response[data[:, f_ind] < f_val]) * (1 - np.mean(response[data[:, f_ind] < f_val])))**2
            large_prop_sum = (np.mean(response[data[:, f_ind] >= f_val]) * (1 - np.mean(response[data[:, f_ind] >= f_val])))** 2
            gini_lv1 = small_prop_sum + large_prop_sum
            gini_lv1_list.append((f_ind, f_val, gini_lv1))
        # This condition takes care if gini_lv1_list is empty which could happen because of
        # the < 5 condition when the dataset is very small. In this case it will take a random value.
        if len(gini_lv1_list) == 0:
            gini_lv1_list.append((f_ind, np.random.choice(rand_feat_value, 1), 0.9999999999))
        gini_lv1_best = min(gini_lv1_list, key = itemgetter(2))
        gini_lv2_list.append(gini_lv1_best)

    return min(gini_lv2_list, key = itemgetter(2))[0:2]
    # This will return the feature index and feature value, respectively

# Initializing accuracy storage lists
avg_acc = list()
fold_acc = list()

# Implementing the new GINI function and calculating accuracies
for iter in range(100):

    for train, test in folds.split(X):
        train_data, test_data = full_data[train], full_data[test]

        tree_model = greedy_tree(train_data, no_of_features = 30, no_of_feat_val = 5,
                                 leaf_min_rec = 20, gini_index = gini_index_2)
        y_pred = []
        for row in test_data[:, :-1]:
            y_pred.append(tree_predict(tree_model, row))
        y_pred = np.array(y_pred)

        fold_acc.append(accuracy_score(test_data[:,-1], y_pred))

    avg_acc.append(np.mean(fold_acc))
    print("Done with iteration: {}. Average accuracy score for iteration: {}".format(iter, np.mean(fold_acc)))

# Calculating the mean and standard deviation of the predictions

cv_mean_2 = np.mean(avg_acc)
cv_sd_2 = np.std(avg_acc)

print("The mean of the accuracies is:", cv_mean_2)
print("The standard deviation of the accuracies is:", cv_sd_2)


# ============================================== Q1 END ===============================================================

# Question 2
# Implementing Adaboost

def adaboost(tree_algo, train_data, no_of_trees = 20):

    # Initializing actual weights
    weights_actual = np.repeat(1/train_data.shape[0], train_data.shape[0])
    # Here at initialization, the probabilistic weights will be equal to actual weights
    weights_prob = weights_actual

    # To calculate probabilistic weights from actual, we will need a function that converts numbers to probs

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # We also need to store the models and stage values
    output_list = []

    for tree in range(no_of_trees):

        # Using weights to adjust the training data - resampling with weights
        mask = np.random.choice(train_data.shape[0], train_data.shape[0], replace=True, p = weights_prob)
        adjusted_data = train_data[mask, :]

        tree_model = tree_algo(adjusted_data, no_of_features=10, no_of_feat_val=5,
                               leaf_min_rec=20, gini_index = gini_index_2)
        y_pred = []
        for row in adjusted_data[:, :-1]:
            y_pred.append(tree_predict(tree_model, row))

        y_pred = np.array(y_pred)

        t_error = abs(adjusted_data[:, -1] - y_pred)

        # If the trees converge very early due to overfitting, this condition will ignore that tree
        # and build the next tree with the same parameters and weights
        if sum(t_error) == 0:
            continue

        error = sum(t_error * weights_actual)/sum(weights_actual)

        stage = np.log((1 - error)/error)

        # Now we will recompute the weights
        weights_actual = weights_actual * np.exp(stage * t_error)
        weights_prob = softmax(weights_actual)

        output_list.append((tree_model, stage))


    return output_list

# Trying out a single run of the function and making prediction on the complete data
ada_models = adaboost(greedy_tree, full_data)

preds_list = []
stages_sum = 0
for model in ada_models:
    preds = []
    for row in X:
        preds.append(tree_predict(model[0], row))
    # Converting list to an array
    preds = np.array(preds)
    preds_list.append(preds * model[1])
    stages_sum += model[1]

preds_list_sum = np.zeros(207)

for i in preds_list:
    preds_list_sum += i

final_pred = (preds_list_sum/stages_sum > 0.5)

print("Accuracy on the complete data is:", accuracy_score(Y, final_pred))

# ================================================ PART C ==============================================================
# Initializing accuracy lists
avg_acc = list()
fold_acc = list()

# Running the algorithm for 100 iterations
for iter in range(100):

    for train, test in folds.split(X):
        train_data, test_data = full_data[train], full_data[test]

        ada_models = adaboost(greedy_tree, train_data)

        preds_list = []
        stages_sum = 0
        for model in ada_models:
            preds = []
            for row in test_data[:, :-1]:
                preds.append(tree_predict(model[0], row))
            # Converting list to an array
            preds = np.array(preds)
            preds_list.append(preds * model[1])
            stages_sum += model[1]

        preds_list_sum = np.zeros(len(test_data))

        for i in preds_list:
            preds_list_sum += i

        final_pred = (preds_list_sum / stages_sum > 0.5)

        fold_acc.append(accuracy_score(test_data[:,-1], final_pred))

    avg_acc.append(np.mean(fold_acc))
    print("Done with iteration: {}. Average accuracy score for iteration: {}".format(iter, np.mean(fold_acc)))


cv_mean_3 = np.mean(avg_acc)
cv_sd_3 = np.std(avg_acc)

print("The mean of the accuracies is:", cv_mean_3)
print("The standard deviation of the accuracies is:", cv_sd_3)


# Part D

# Plotting the test and train error out based on the number of weak learners

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

no_weak_learners = [0,1,5,10,15,20,25,30,35,40]

# Initializing the lists to store the errors
train_error = []
test_error = []

for trees in no_weak_learners:
    ada_model = adaboost(greedy_tree, train_data)

    # Predicting on train

    preds_list = []
    stages_sum = 0
    for model in ada_model:
        preds = []
        for row in X_train:
            preds.append(tree_predict(model[0], row))
        # Converting list to an array
        preds = np.array(preds)
        preds_list.append(preds * model[1])
        stages_sum += model[1]

    preds_list_sum = np.zeros(X_train.shape[0])

    for i in preds_list:
        preds_list_sum += i

    final_pred = (preds_list_sum / stages_sum > 0.5)

    train_error.append(accuracy_score(Y_train, final_pred))

    # Predicting on test

    preds_list = []
    stages_sum = 0
    for model in ada_model:
        preds = []
        for row in X_test:
            preds.append(tree_predict(model[0], row))
        # Converting list to an array
        preds = np.array(preds)
        preds_list.append(preds * model[1])
        stages_sum += model[1]

    preds_list_sum = np.zeros(X_test.shape[0])

    for i in preds_list:
        preds_list_sum += i

    final_pred = (preds_list_sum / stages_sum > 0.5)

    train_error.append(accuracy_score(Y_test, final_pred))


# Plotting the graph

import matplotlib.pyplot as plt

plt.plot(no_weak_learners, list(train_error))
plt.plot(no_weak_learners, list(test_error))

plt.legend(['Training error', 'Test error'], loc='upper right')
plt.title('Error improvement with increasing number of weak learners')
plt.xlabel('Number of weak learners in Adaboost')
plt.ylabel('Error')

plt.show()

#======================================================================================================================