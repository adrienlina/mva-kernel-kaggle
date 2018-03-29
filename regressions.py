import numpy as np

from data_manipulation import get_precision


REG_PARAMS_SPAN = [10**i for i in range(-10, 10)] + [10**i/2 for i in range(-10, 10)]


def ridge_regression(K, y, reg):
    """
    Compute alpha for a ridge regression as explained in course slides p.90-92

    PARAMETERS:
    - K (n,n): K(x,x) of the train data points
    - y (n,): train labels (0 or 1)
    - reg: the regularization parameter

    RETURNS:
    - alpha of size (n,)
    """
    n = y.shape[0]

    return np.linalg.inv(K + n * reg * np.eye(n)) @ y


def get_ridge_prediction(K_x, alpha):
    """
    Get ridge regression prediction

    PARAMETERS:
    - K (n1,n2): K(x1,x2) with x1 the train data points and x2 the test data points
    - alpha (n,): the coefficient for each point, as explained in the course
        slides p.90

    RETURNS:
    - (0-1) array of size (m,)
    """
    proba = alpha @ K_x

    return (proba >= 1/2).astype(int)


def get_best_reg_param(kernel, regression, data):
    """
    Find the best regularization parameter for a given kernel, regression regression
    and a (train_X, train_y, test_X, test_y) dataset

    PARAMETERS:
    - kernel: a kernel to apply
    - regression: a regression method to produce alpha
    - data: a 4-uple containing in order:
        - train_X (n,d): a training set of points
        - train_y (n,): a training set of labels
        - test_X (m,d): a test set of points
        - test_y (m,): a test set of labels

    RETURNS:
    - 2-uple of the best parameter and the associated test precision obtained.
    """
    train_X, train_y, test_X, test_y = data

    K = kernel(train_X, train_X)
    K_x = kernel(train_X, test_X)

    test_precisions = []

    for reg in REG_PARAMS_SPAN:
        alpha = regression(K, train_y, reg)
        pred = get_ridge_prediction(K_x, alpha)
        test_precisions.append(get_precision(pred, test_y))

    best_reg_index = max(range(len(REG_PARAMS_SPAN)), key=lambda x: test_precisions[x])
    return REG_PARAMS_SPAN[best_reg_index], test_precisions[best_reg_index]
