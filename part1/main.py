import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *

#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################

# TODO: first fill out functions in linear_regression.py, otherwise the functions below will not work


def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error


# Don't run this until the relevant functions in linear_regression.py have been fully implemented.
print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=1))
print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=0.1))
print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=0.01))

#######################################################################
# 3. Support Vector Machine
#######################################################################

# TODO: first fill out functions in svm.py, or the functions below will not work

def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())


def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################

# TODO: first fill out functions in softmax.py, or run_softmax_on_MNIST will not work


def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")

    # Update the labels to mod 3
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)
    test_error_mod3 = compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter)
    print(f'softmax test_error_mod3 = {test_error_mod3}')
    return test_error


print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))
# TODO: Find the error rate for temp_parameter = [.5, 1.0, 2.0]
#      Remember to return the tempParameter to 1, and re-run run_softmax_on_MNIST

#######################################################################
# 6. Changing Labels
#######################################################################



def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()

    # Update the labels to mod 3
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)

    # Train the model with the new labels
    theta, cost_function_history = softmax_regression(train_x, train_y_mod3, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=3, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)

    # Compute the test error with the new labels
    test_error_mod3 = compute_test_error(test_x, test_y_mod3, theta, temp_parameter)
    print(f'softmax test_error_mod3 = {test_error_mod3}')

    return test_error_mod3

# Run the function and report the error rate
print('Error rate when trained on labels mod 3:', run_softmax_on_MNIST_mod3(temp_parameter=1))




#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################


## Dimensionality reduction via PCA ##

# TODO: First fill out the PCA functions in features.py as the below code depends on them.


n_components = 18

###Correction note:  the following 4 lines have been modified since release.
train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)
# Project the training and test data onto the first 10 principal components
train_pca10 = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca10 = project_onto_PC(test_x, pcs, n_components, feature_means)

# Apply the cubic feature mapping to the 10-dimensional PCA representations
train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)

# train_pca (and test_pca) is a representation of our training (and test) data
# after projecting each example onto the first 18 principal components.


# TODO: Train your softmax regression model using (train_pca, train_y)
#       and evaluate its accuracy on (test_pca, test_y).
# Train the softmax regression model using the cubic kernel-transformed data
theta, cost_function_history = softmax_regression(train_cube, train_y, temp_parameter=1, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
plot_cost_function_over_time(cost_function_history)


# Compute the test error using the cubic kernel-transformed test data
test_error_cube = compute_test_error(test_cube, test_y, theta, temp_parameter=1)
print("Error rate for 10-dimensional cubic PCA features =", test_error_cube)

# TODO: Use the plot_PC function in features.py to produce scatterplot
#       of the first 100 MNIST images, as represented in the space spanned by the
#       first 2 principal components found above.
plot_PC(train_x[range(000, 100), ], pcs, train_y[range(000, 100)], feature_means)#feature_means added since release


# TODO: Use the reconstruct_PC function in features.py to show
#       the first and second MNIST images as reconstructed solely from
#       their 18-dimensional principal component representation.
#       Compare the reconstructed images with the originals.
firstimage_reconstructed = reconstruct_PC(train_pca10[0, ], pcs, n_components, train_x, feature_means)#feature_means added since release
plot_images(firstimage_reconstructed)
plot_images(train_x[0, ])

secondimage_reconstructed = reconstruct_PC(train_pca10[1, ], pcs, n_components, train_x, feature_means)#feature_means added since release
plot_images(secondimage_reconstructed)
plot_images(train_x[1, ])


## Cubic Kernel ##
# TODO: Find the 10-dimensional PCA representation of the training and test set

# Find the 10-dimensional PCA representation of the training and test set
n_components = 10

# Center the training data
train_x_centered, feature_means = center_data(train_x)

# Compute the principal components
pcs = principal_components(train_x_centered)

# Project the training and test data onto the first 10 principal components
train_pca10 = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca10 = project_onto_PC(test_x, pcs, n_components, feature_means)
# TODO: First fill out cubicFeatures() function in features.py as the below code requires it.

train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)
# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.


# TODO: Train your softmax regression model using (train_cube, train_y)
#       and evaluate its accuracy on (test_cube, test_y).
# Train the softmax regression model using the cubic kernel-transformed data
theta, cost_function_history = softmax_regression(train_cube, train_y, temp_parameter=1, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
plot_cost_function_over_time(cost_function_history)

# Compute the test error using the cubic kernel-transformed test data
test_error_cube = compute_test_error(test_cube, test_y, theta, temp_parameter=1)
print("Error rate for cubic kernel features =", test_error_cube)

# Train the SVM model using the cubic polynomial kernel on the 10-dimensional PCA representations
svm_model = SVC(kernel='poly', degree=3, random_state=0)
svm_model.fit(train_pca10, train_y)

# Evaluate the model's accuracy on the test set
test_predictions = svm_model.predict(test_pca10)
test_accuracy = accuracy_score(test_y, test_predictions)
test_error = 1 - test_accuracy

# correct value 0.07640000000000002
print("Error rate for 10-dimensional PCA features using cubic polynomial SVM =", test_error)

# Train the SVM model using the RBF kernel on the 10-dimensional PCA representations
svm_model = SVC(kernel='rbf', random_state=0)
svm_model.fit(train_pca10, train_y)

# Evaluate the model's accuracy on the test set
test_predictions = svm_model.predict(test_pca10)
test_accuracy = accuracy_score(test_y, test_predictions)
test_error = 1 - test_accuracy

# Apply the RBF SVM to the 10-dimensional PCA representation of the training data
svm_rbf = SVC(kernel='rbf', random_state=0)
svm_rbf.fit(train_pca10, train_y)
rbf_svm_predictions = svm_rbf.predict(test_pca10)
# Calculate the error rate
rbf_svm_test_error = 1 - accuracy_score(test_y, rbf_svm_predictions)
#correct value 0.06359999999999999
print(f"Error rate for 10-dimensional PCA features using RBF SVM: {rbf_svm_test_error}")